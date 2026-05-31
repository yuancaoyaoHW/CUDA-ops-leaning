# DPO（Direct Preference Optimization）

## 1. 学习目标

- 理解 DPO 的方法动机：绕过显式 reward model 直接从偏好数据优化策略
- 掌握 DPO 的数学推导：从 RLHF objective 到 closed-form solution
- 理解 reference model 的作用与 β 参数的影响
- 能够配置和执行 DPO 训练
- 掌握 DPO 的常见失败模式与调优策略

## 2. 方法动机

### 2.1 RLHF 的复杂性

标准 RLHF pipeline：
```
1. 训练 Reward Model（需要偏好数据）
2. 用 PPO 优化 policy（需要 4 个模型：policy, ref, reward, critic）
3. 复杂的超参数调优（KL penalty, clip ratio, GAE lambda...）
```

DPO 的简化：
```
1. 直接从偏好数据优化 policy（只需 2 个模型：policy, reference）
2. 无需训练 reward model
3. 无需 PPO 的复杂训练循环
4. 等价于隐式地优化了一个 reward function
```

### 2.2 核心洞察

DPO 证明了：在 KL-constrained reward maximization 问题中，最优策略有 closed-form 解：
```
π*(y|x) = (1/Z(x)) × π_ref(y|x) × exp(r(x,y) / β)
```

反解 reward：
```
r(x,y) = β × log(π*(y|x) / π_ref(y|x)) + β × log Z(x)
```

将这个 reward 代入 Bradley-Terry preference model，得到 DPO loss。

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| DPO | Direct Preference Optimization | 直接偏好优化，无需 reward model 的对齐方法 |
| Reference Model | Reference Model | 冻结的 SFT 模型，作为 KL 约束的锚点 |
| Preference Pair | Preference Pair | (prompt, chosen, rejected) 三元组 |
| Bradley-Terry | Bradley-Terry Model | 偏好概率模型：P(y_w > y_l) = σ(r_w - r_l) |
| β (beta) | Temperature Parameter | 控制偏离 reference model 的程度 |
| Implicit Reward | Implicit Reward | DPO 隐式定义的 reward function |
| Chosen | Chosen Response | 偏好数据中被选择的（更好的）回答 |
| Rejected | Rejected Response | 偏好数据中被拒绝的（更差的）回答 |

## 4. 数学定义

### 4.1 RLHF Objective

```
max_π E_{x~D, y~π(·|x)} [r(x,y)] - β × KL(π || π_ref)
```

### 4.2 最优策略（closed-form）

```
π*(y|x) = π_ref(y|x) × exp(r(x,y) / β) / Z(x)
```

### 4.3 隐式 Reward

```
r(x,y) = β × log(π_θ(y|x) / π_ref(y|x)) + β × log Z(x)
```

### 4.4 DPO Loss

```
L_DPO(θ) = -E_{(x, y_w, y_l) ~ D} [log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

简写：
L_DPO = -E [log σ(β × (r_w - r_l))]

其中：
r_w = log π_θ(y_w|x) - log π_ref(y_w|x)  (chosen 的 implicit reward)
r_l = log π_θ(y_l|x) - log π_ref(y_l|x)  (rejected 的 implicit reward)
```

## 5. Loss 推导

### 5.1 从 Bradley-Terry 到 DPO

Bradley-Terry preference model：
```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

将隐式 reward 代入：
```
P(y_w > y_l | x) = σ(β × [log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)])
```

注意 Z(x) 项在相减时消掉了！

最大化 log-likelihood：
```
L = -log P(y_w > y_l | x)
  = -log σ(β × [log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)])
```

### 5.2 梯度分析

```
∇L = -β × σ(-β×Δ) × [∇log π_θ(y_w|x) - ∇log π_θ(y_l|x)]

其中 Δ = log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)

解读：
- σ(-β×Δ) 是"错误程度"：模型越分不清 chosen/rejected，梯度越大
- ∇log π_θ(y_w|x)：增加 chosen 的概率
- -∇log π_θ(y_l|x)：降低 rejected 的概率
```

## 6. 数据格式

```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "chosen": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously...",
  "rejected": "Quantum computing is very complex and involves many mathematical concepts that are difficult to explain simply..."
}
```

### 数据质量要求

- Chosen 和 rejected 应该是对同一 prompt 的不同回答
- 质量差异应该明显但不极端
- 避免 length bias（chosen 不应总是更长）
- 多样化的 prompt 分布

## 7. 训练配置

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,          # DPO 学习率通常很小
    beta=0.1,                     # KL penalty 强度
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    max_length=2048,
    max_prompt_length=1024,
    gradient_checkpointing=True,
    loss_type="sigmoid",          # 标准 DPO loss
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,          # 冻结的 SFT 模型
    args=config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
```

## 8. 显存估算

```
DPO 需要同时加载 2 个模型：
- Policy model: P × 2 bytes (BF16)
- Reference model: P × 2 bytes (BF16, frozen)
- Gradients (policy only): P × 2 bytes
- Optimizer (policy only): P × 8 bytes (AdamW)

8B model DPO:
- Policy: 16 GB
- Reference: 16 GB
- Gradients: 16 GB
- Optimizer: 64 GB
Total: ~112 GB → 需要多卡或 LoRA

With LoRA:
- Policy (frozen + LoRA): 16 GB + 0.2 GB
- Reference: 16 GB
- LoRA gradients: 0.2 GB
- LoRA optimizer: 0.8 GB
Total: ~33 GB → 单卡 A100 可行
```

## 9. 吞吐瓶颈

- **Forward pass ×4**：每个 batch 需要 policy(chosen), policy(rejected), ref(chosen), ref(rejected)
- **显存**：两个模型同时在 GPU 上
- **数据**：偏好数据通常较少（几万到几十万条）

## 10-20. 关键内容

### β 参数的影响

| β 值 | 效果 | 风险 |
|------|------|------|
| 0.01 | 强烈偏离 reference | 过拟合偏好数据，忘记通用能力 |
| 0.1 | 标准设置 | 平衡 |
| 0.5 | 保守，接近 reference | 学习不充分 |
| 1.0 | 几乎不学习 | 浪费训练 |

### 常见失败模式

1. **Reward hacking**：模型学会了 chosen/rejected 的表面特征（如长度）而非质量
2. **KL explosion**：β 太小，模型偏离 reference 太远
3. **Loss 不下降**：学习率太小或数据质量差
4. **Chosen 概率下降**：known issue，DPO 可能同时降低 chosen 和 rejected 的概率
5. **Length bias**：模型学会生成更长/更短的回答

### 习题（选 5 道）

1. 写出 DPO loss 的完整公式并解释每一项的含义。
2. DPO 的 reference model 起什么作用？如果去掉会怎样？
3. β=0.1 和 β=0.5 的区别是什么？如何选择？
4. DPO 相比 PPO 的优势和劣势分别是什么？
5. 如何检测 DPO 训练中的 reward hacking？

### 复习卡片（选 10 张）

1. Q: DPO loss 公式？ A: L = -log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
2. Q: DPO 需要几个模型？ A: 2 个（policy + frozen reference）
3. Q: β 的作用？ A: 控制偏离 reference 的程度，越小越激进
4. Q: DPO 的隐式 reward？ A: r(x,y) = β × log(π_θ(y|x) / π_ref(y|x))
5. Q: DPO vs PPO 的核心区别？ A: DPO 无需显式 reward model 和 RL 训练循环
6. Q: DPO 的典型学习率？ A: 1e-7 ~ 5e-7（比 SFT 小 10-100x）
7. Q: DPO 的典型 β？ A: 0.1（标准）
8. Q: DPO 数据格式？ A: (prompt, chosen, rejected) 三元组
9. Q: DPO 的 forward pass 次数？ A: 4 次（policy×2 + ref×2）
10. Q: DPO chosen 概率下降问题？ A: Known issue，可用 IPO/cDPO 缓解
