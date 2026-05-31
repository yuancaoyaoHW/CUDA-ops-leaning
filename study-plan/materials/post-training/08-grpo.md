# GRPO（Group Relative Policy Optimization）

## 1. 学习目标

- 理解 GRPO 的核心创新：无 Critic、组内相对优势估计
- 掌握 GRPO 与 PPO 的数学差异
- 理解 DeepSeek 在 GRPO 上的工程实践
- 能够分析 GRPO 的显存优势与适用场景
- 掌握 GRPO 的 group size 选择与 baseline 设计

## 2. 方法动机

### 2.1 PPO 的问题

- 需要训练 Critic model → 额外显存和计算
- Critic 的 value estimation 不准确 → 高方差
- 4 个模型同时在 GPU → 显存压力大
- 训练不稳定，超参数敏感

### 2.2 GRPO 的简化

核心思想：**用同一 prompt 的多个 rollout 的相对 reward 作为 baseline，替代 Critic**

```
PPO:  Advantage = Reward - V(s)     (需要 Critic 估计 V)
GRPO: Advantage = (Reward - mean(group_rewards)) / std(group_rewards)
```

优势：
- 无需 Critic model → 节省 ~50% 显存
- 无 value estimation error → 更稳定
- 实现更简单
- DeepSeek-R1 验证了大规模有效性

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| GRPO | Group Relative Policy Optimization | 组相对策略优化 |
| Group | Group | 同一 prompt 的多个 rollout 组成的组 |
| Group Size | Group Size (G) | 每个 prompt 生成的回答数量 |
| Relative Advantage | Relative Advantage | 组内标准化后的 reward 作为 advantage |
| Baseline | Baseline | 组内 reward 的均值，替代 Critic |

## 4. 数学定义

### 4.1 GRPO Objective

```
L_GRPO = E_{x~D} [1/G × Σ_{i=1}^{G} min(r_i × Â_i, clip(r_i, 1-ε, 1+ε) × Â_i)]
       - β × KL(π_θ || π_ref)

其中：
r_i = π_θ(y_i|x) / π_old(y_i|x)
Â_i = (R_i - mean(R_1,...,R_G)) / std(R_1,...,R_G)  (组内标准化)
G = group size
```

### 4.2 与 PPO 的对比

```
PPO:  A_t = R_t + γV(s_{t+1}) - V(s_t)  → 需要 Critic
GRPO: Â_i = (R_i - μ_group) / σ_group    → 只需 group statistics
```

### 4.3 KL Penalty

```
KL = E_{y~π_θ} [log π_θ(y|x) - log π_ref(y|x)]

实现方式：
- Per-token KL（更精确）
- Sequence-level KL（更简单）
```

## 5. Loss 推导

### 5.1 为什么 Group Relative 有效？

直觉：对于同一个 prompt，如果生成了 G 个回答：
- Reward 高于平均的回答 → 正 advantage → 增加概率
- Reward 低于平均的回答 → 负 advantage → 降低概率

这等价于一个**无偏的 baseline**：
```
E[Â_i] = E[(R_i - μ) / σ] = 0  (无偏)
Var[Â_i] = 1  (标准化，稳定训练)
```

### 5.2 Group Size 的影响

```
G=2:  最小组，方差估计不准，但效率高
G=4:  常用设置，平衡精度和效率
G=8:  更准确的 baseline，但 rollout 成本高
G=16: DeepSeek 使用，大规模训练
G=64: 极端设置，baseline 非常准确
```

## 6. 数据格式

与 PPO 相同，只需要 prompts：
```json
{"prompt": "Solve: What is 15 × 23?"}
{"prompt": "Write a function to check if a number is prime."}
```

GRPO 特别适合有 verifiable reward 的场景（数学、代码）。

## 7. 训练配置

```python
# GRPO 配置（基于 trl 或自定义实现）
config = {
    "group_size": 8,              # 每个 prompt 生成 8 个回答
    "learning_rate": 1e-6,
    "kl_coef": 0.05,              # β
    "clip_range": 0.2,            # ε
    "num_train_epochs": 1,
    "temperature": 1.0,           # rollout 采样温度
    "max_new_tokens": 512,
    "normalize_advantage": True,  # 组内标准化
    "whiten_rewards": False,      # 是否全局标准化 reward
}
```

## 8. 显存估算

```
GRPO (8B model, BF16):
- Actor (trainable): 16 + 16 + 64 = 96 GB
- Reference (frozen): 16 GB
- NO Critic needed!

Total: ~112 GB (vs PPO ~224 GB)
节省: ~50%

With LoRA:
- Actor (frozen + LoRA): 16 + 0.2 + 0.2 + 0.8 = 17.2 GB
- Reference: 16 GB
Total: ~33 GB → 单卡可行
```

## 9. 吞吐瓶颈

```
GRPO 的主要开销：
- Rollout: G 个回答 per prompt → G× decode 时间
- Reward: G 个回答需要评分
- Training: 标准 PPO update（但无 Critic）

Rollout 仍是最大瓶颈（G=8 意味着 8× 生成量）

优化：
- 使用 vLLM/SGLang 高效 batch generation
- 减小 G（但会降低 baseline 质量）
- 并行 rollout + training（pipeline）
```

## 10-20. 关键内容

### DeepSeek 的 GRPO 实践

DeepSeek-R1 使用 GRPO 训练推理能力：
- Group size: 16-64
- Reward: rule-based verifier（数学正确性、代码通过率）
- 无需人工标注的 reward model
- 大规模训练（数千 GPU）
- 结果：超越 PPO 的推理能力

### GRPO vs PPO vs DPO 对比

| 维度 | PPO | GRPO | DPO |
|------|-----|------|-----|
| 需要 Critic | ✓ | ✗ | ✗ |
| 需要 Reward Model | ✓ | ✓ (或 verifier) | ✗ |
| 需要 Reference | ✓ | ✓ | ✓ |
| Online 生成 | ✓ | ✓ | ✗ |
| 显存 (8B) | ~224 GB | ~112 GB | ~112 GB |
| 训练稳定性 | 低 | 中 | 高 |
| 适合 verifiable tasks | ✓ | ✓✓ | ✗ |
| 实现复杂度 | 高 | 中 | 低 |

### 习题（选 5 道）

1. GRPO 如何替代 PPO 中的 Critic？数学上等价吗？
2. Group size G 的选择如何影响训练效果和效率？
3. 为什么 GRPO 特别适合有 verifiable reward 的任务？
4. GRPO 的 advantage 标准化为什么重要？如果不标准化会怎样？
5. DeepSeek-R1 使用 GRPO 的关键设计决策是什么？
