# PPO（Proximal Policy Optimization）

## 1. 学习目标

- 理解 PPO 在 RLHF 中的角色与训练流程
- 掌握 PPO 的 clipped objective 与 GAE 优势估计
- 理解 PPO 训练中 4 个模型的协作关系
- 能够分析 PPO 训练的显存需求与吞吐瓶颈
- 掌握 PPO 的常见失败模式（reward hacking、KL explosion）

## 2. 方法动机

PPO 是 RLHF 的经典 RL 算法，用于根据 reward model 的信号优化 LLM 策略。

### 2.1 RLHF Pipeline

```
Stage 1: SFT → 基础对话能力
Stage 2: Reward Model Training → 学习人类偏好
Stage 3: PPO → 用 RM 信号优化策略
```

### 2.2 PPO 的 4 个模型

| 模型 | 作用 | 是否更新 | 显存 |
|------|------|---------|------|
| Policy (Actor) | 生成回答 | ✓ | weights + grads + optimizer |
| Reference | KL 约束锚点 | ✗ | weights only |
| Reward Model | 评分 | ✗ | weights only |
| Critic (Value) | 估计 V(s) | ✓ | weights + grads + optimizer |

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| PPO | Proximal Policy Optimization | 近端策略优化 |
| Actor | Actor / Policy | 生成回答的模型 |
| Critic | Critic / Value Model | 估计状态价值的模型 |
| GAE | Generalized Advantage Estimation | 广义优势估计 |
| Clip Ratio | Clip Ratio (ε) | PPO 的截断比率（通常 0.2） |
| KL Penalty | KL Penalty | 防止策略偏离 reference 的惩罚 |
| Rollout | Rollout | 用当前策略生成回答的过程 |
| Reward Shaping | Reward Shaping | 将 RM score 转化为 token-level reward |

## 4. 数学定义

### 4.1 PPO Objective

```
L_PPO = E_t [min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]

其中：
r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)  (importance sampling ratio)
A_t = advantage at time t
ε = clip ratio (通常 0.2)
```

### 4.2 GAE (Generalized Advantage Estimation)

```
A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l × δ_{t+l}

δ_t = r_t + γ × V(s_{t+1}) - V(s_t)  (TD error)

γ = discount factor (通常 1.0 for LLM)
λ = GAE lambda (通常 0.95)
```

### 4.3 KL Penalty

```
reward_total = reward_RM - β × KL(π_θ || π_ref)

KL per token: KL_t = log π_θ(a_t|s_t) - log π_ref(a_t|s_t)
```

### 4.4 Value Loss

```
L_value = E_t [(V_θ(s_t) - V_target_t)²]

V_target_t = A_t^GAE + V_old(s_t)  (returns)
```

## 5. Loss 推导

### 5.1 为什么需要 Clip？

不 clip 的 policy gradient：
```
L = E[r_t × A_t]
```
问题：r_t 可能很大（策略变化太多）→ 训练不稳定

Clip 的作用：
```
当 A_t > 0（好动作）：r_t 被 clip 到 [1-ε, 1+ε] → 限制概率增加幅度
当 A_t < 0（坏动作）：r_t 被 clip 到 [1-ε, 1+ε] → 限制概率减少幅度
```

### 5.2 RLHF 中的 Reward Shaping

RM 只给整个回答一个 score，但 PPO 需要 token-level reward：
```
r_t = 0                           for t < T (中间 token)
r_T = RM_score - β × KL_total    for t = T (最后一个 token)

或者 per-token KL penalty：
r_t = -β × KL_t                  for t < T
r_T = RM_score - β × KL_T        for t = T
```

## 6. 数据格式

PPO 不需要偏好数据，只需要 prompts：
```json
{"prompt": "Explain quantum computing in simple terms."}
{"prompt": "Write a Python function to sort a list."}
{"prompt": "What are the benefits of exercise?"}
```

训练过程中动态生成 responses 并用 RM 评分。

## 7. 训练配置

```python
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name="meta-llama/Llama-3-8B-sft",
    learning_rate=1e-6,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,              # 每批数据训练几轮
    gradient_accumulation_steps=8,
    kl_penalty="kl",           # "kl" or "abs" or "mse"
    init_kl_coef=0.1,          # β 初始值
    target_kl=6.0,             # 目标 KL，超过则增大 β
    clip_range=0.2,            # ε
    vf_coef=0.1,               # value loss 系数
    gamma=1.0,                 # discount factor
    lam=0.95,                  # GAE lambda
    max_grad_norm=1.0,
)
```

## 8. 显存估算

```
8B model PPO (BF16):
- Actor (trainable): 16 + 16 + 64 = 96 GB (weights + grads + optimizer)
- Reference (frozen): 16 GB
- Reward Model (frozen): 16 GB (可与 ref 共享架构)
- Critic (trainable): 16 + 16 + 64 = 96 GB (通常与 actor 共享 backbone)

Total naive: ~224 GB → 需要多卡

优化方案：
- Actor-Critic 共享 backbone: 节省 ~80 GB
- LoRA actor + critic: 大幅减少 trainable params
- Offload reference/RM to CPU: 节省 32 GB GPU
- DeepSpeed ZeRO-3: 分布式显存
```

## 9. 吞吐瓶颈

PPO 训练的主要瓶颈：

| 阶段 | 操作 | 瓶颈 | 占比 |
|------|------|------|------|
| Rollout | 用 actor 生成回答 | Decode latency | 60-80% |
| Reward | RM forward pass | Compute | 5-10% |
| Reference | Ref model forward | Compute | 5-10% |
| PPO Update | Actor + Critic backward | Compute + Memory | 10-20% |

**Rollout 是最大瓶颈**：生成 64 个回答（每个 256 tokens）需要大量 decode 时间。

优化：
- 使用 vLLM/SGLang 作为 rollout engine（高效 batched decode）
- Async rollout（生成和训练重叠）
- 减少 generation length

## 10-20. 关键内容

### 常见失败模式

1. **Reward Hacking**：模型找到 RM 的漏洞（如生成特定格式获得高分但内容差）
2. **KL Explosion**：策略偏离 reference 太远，生成质量崩溃
3. **Reward Collapse**：所有回答得到相似 reward，梯度消失
4. **Value Loss 不收敛**：Critic 无法准确估计 value
5. **Mode Collapse**：模型只生成少数几种回答

### Debug Checklist

- [ ] 监控 KL divergence（应在 target_kl 附近）
- [ ] 监控 reward 分布（应有方差，不应全部相同）
- [ ] 检查 clip fraction（应在 10-30%，太高说明步长太大）
- [ ] 验证 rollout 质量（人工检查生成的回答）
- [ ] 监控 value loss（应稳定下降）
- [ ] 检查 advantage 的均值和方差
- [ ] 对比 actor 和 reference 的 KL
- [ ] 检查 reward model 的 calibration

### 习题（选 5 道）

1. PPO 的 clip objective 解决了什么问题？ε=0.2 意味着什么？
2. 为什么 RLHF 中的 PPO 需要 KL penalty？如果不加会怎样？
3. GAE 中 λ=0 和 λ=1 分别对应什么？为什么选 0.95？
4. PPO 训练中 rollout 为什么是最大瓶颈？如何优化？
5. 如何检测 reward hacking？有哪些防御策略？
