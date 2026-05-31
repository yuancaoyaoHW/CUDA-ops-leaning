# 02 - LoRA 与 QLoRA

## 1. 学习目标

- 理解 LoRA（Low-Rank Adaptation）的低秩分解原理及其在大模型微调中的优势
- 掌握 rank 选择、target modules 选择、alpha 参数调节的工程实践
- 理解 QLoRA 的 NF4（NormalFloat 4-bit）量化原理
- 掌握 double quantization、paged optimizer 的内存优化机制
- 能够在单卡 24GB 显存上完成 7B-70B 模型的高效微调
- 理解 LoRA 与 full fine-tuning 的性能差距及适用场景

## 2. 方法动机

大语言模型参数量从 7B 到 405B 不等，全参数微调（Full Fine-tuning）需要巨大的显存和计算资源。以 LLaMA-2 70B 为例，仅模型参数就需要 140GB（FP16），加上优化器状态和梯度，单次训练需要数百 GB 显存。

LoRA 的核心洞察：预训练模型的权重更新矩阵 ΔW 具有低秩特性（low intrinsic dimensionality）。因此可以将 ΔW 分解为两个低秩矩阵的乘积，大幅减少可训练参数量。

QLoRA 进一步将基础模型量化到 4-bit，结合 LoRA 的低秩适配器，实现了在单张 48GB GPU 上微调 65B 模型的突破。

**关键优势：**
- 参数效率：仅训练 0.1%-1% 的参数
- 显存节省：4-bit 量化 + 低秩适配器
- 无推理延迟：LoRA 权重可合并回原模型
- 模块化：不同任务的 LoRA 可热切换

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| LoRA | Low-Rank Adaptation | 通过低秩矩阵分解实现参数高效微调的方法 |
| QLoRA | Quantized Low-Rank Adaptation | 结合 4-bit 量化与 LoRA 的高效微调方法 |
| Rank (r) | Rank | 低秩分解的秩，控制适配器容量 |
| Alpha (α) | Scaling factor | LoRA 的缩放系数，控制更新幅度 |
| NF4 | NormalFloat 4-bit | 基于正态分布的 4-bit 量化格式 |
| Double Quantization | Double Quantization | 对量化常数再次量化以节省显存 |
| Paged Optimizer | Paged Optimizer | 利用 CPU 内存分页管理优化器状态 |
| Target Modules | Target Modules | 应用 LoRA 的目标线性层 |
| Intrinsic Dimension | Intrinsic Dimensionality | 模型有效参数空间的内在维度 |
| Adapter | Adapter | 插入预训练模型的可训练小模块 |

## 4. 数学定义

### 4.1 LoRA 基本公式

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将权重更新约束为低秩分解：

$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- $B \in \mathbb{R}^{d \times r}$，初始化为零矩阵
- $A \in \mathbb{R}^{r \times k}$，使用 Kaiming 均匀初始化
- $r \ll \min(d, k)$ 为秩

前向传播：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

其中 $\frac{\alpha}{r}$ 为缩放因子（scaling factor），确保不同 rank 下更新幅度一致。

### 4.2 参数量计算

原始参数量：$d \times k$

LoRA 参数量：$r \times (d + k)$

压缩比：$\frac{r(d+k)}{dk}$

以 LLaMA-7B 的 attention 层为例（$d=k=4096, r=16$）：
- 原始：$4096 \times 4096 = 16,777,216$
- LoRA：$16 \times (4096 + 4096) = 131,072$
- 压缩比：0.78%

### 4.3 NF4 量化

NF4 基于假设：预训练权重近似服从正态分布 $\mathcal{N}(0, \sigma^2)$。

量化步骤：
1. 计算分位数（quantile）：将标准正态分布等概率划分为 $2^k = 16$ 个区间
2. 归一化：$w_{norm} = w / \text{absmax}(W_{block})$
3. 映射到最近的量化值：$w_q = \text{argmin}_{q_i} |w_{norm} - q_i|$

反量化：$\hat{w} = q_{w_q} \times \text{absmax}(W_{block})$

### 4.4 Double Quantization

对量化常数（scaling constants）进行二次量化：

- 第一次量化：FP32 → NF4，每 64 个参数共享一个 FP32 scaling constant
- 第二次量化：将 FP32 scaling constants 量化为 FP8，每 256 个 constants 共享一个 FP32 second-level constant

显存节省计算：
- 无 double quant：每个参数额外 $32/64 = 0.5$ bit
- 有 double quant：每个参数额外 $8/64 + 32/(64 \times 256) \approx 0.127$ bit

## 5. Loss 推导

### 5.1 标准语言模型 Loss

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; W_0 + \frac{\alpha}{r}BA)$$

### 5.2 梯度计算

对 $B$ 的梯度：
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T$$

对 $A$ 的梯度：
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T$$

### 5.3 QLoRA 中的梯度

由于基础模型被量化，梯度通过量化权重的直通估计器（Straight-Through Estimator, STE）传播：

$$\frac{\partial \mathcal{L}}{\partial W_0^{NF4}} \approx \frac{\partial \mathcal{L}}{\partial \hat{W}_0}$$

但实际上 QLoRA 中 $W_0$ 被冻结，梯度仅流经 LoRA 分支：

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T, \quad \frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T$$

## 6. 数据格式

### 6.1 指令微调数据格式

```json
{
  "instruction": "将以下英文翻译为中文",
  "input": "The quick brown fox jumps over the lazy dog.",
  "output": "敏捷的棕色狐狸跳过了懒惰的狗。"
}
```

### 6.2 对话格式（ChatML）

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "解释量子计算"},
    {"role": "assistant", "content": "量子计算是利用量子力学原理..."}
  ]
}
```

### 6.3 Tokenization 注意事项

- 仅对 output/assistant 部分计算 loss
- 使用 `labels = -100` 标记不参与 loss 计算的 token
- 确保 special tokens（BOS、EOS、turn markers）正确处理

## 7. 训练配置

### 7.1 LoRA 配置（使用 PEFT 库）

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                          # rank
    lora_alpha=32,                 # scaling factor: alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### 7.2 QLoRA 配置

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

### 7.3 训练超参数

```python
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.001,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
)
```

### 7.4 Rank 选择指南

| 任务复杂度 | 推荐 Rank | 说明 |
|-----------|-----------|------|
| 简单格式适配 | 4-8 | 风格转换、格式调整 |
| 领域知识注入 | 16-32 | 医疗、法律等专业领域 |
| 复杂推理增强 | 64-128 | 数学、代码生成 |
| 接近全参数 | 256+ | 大规模能力迁移 |

## 8. 显存估算

### 8.1 LoRA 显存组成

| 组件 | 计算公式 | 7B 模型 (r=16) |
|------|----------|----------------|
| 基础模型 (FP16) | params × 2 bytes | 14 GB |
| LoRA 参数 (FP16) | lora_params × 2 bytes | ~26 MB |
| 优化器状态 (AdamW) | lora_params × 8 bytes | ~104 MB |
| 梯度 | lora_params × 2 bytes | ~26 MB |
| 激活值 (batch=4, seq=2048) | 取决于 gradient checkpointing | 4-12 GB |
| **总计** | | ~20 GB |

### 8.2 QLoRA 显存组成

| 组件 | 计算公式 | 7B 模型 (r=16) |
|------|----------|----------------|
| 基础模型 (NF4) | params × 0.5 bytes + quant overhead | ~4.2 GB |
| LoRA 参数 (BF16) | lora_params × 2 bytes | ~26 MB |
| 优化器状态 (Paged AdamW) | lora_params × 8 bytes | ~104 MB |
| 梯度 (BF16) | lora_params × 2 bytes | ~26 MB |
| 激活值 | 取决于 batch size | 3-8 GB |
| **总计** | | ~10 GB |

### 8.3 不同模型规模的 QLoRA 显存需求

| 模型规模 | NF4 模型 | LoRA (r=16) | 总显存需求 | 推荐 GPU |
|----------|----------|-------------|-----------|----------|
| 7B | 4.2 GB | ~0.2 GB | ~10 GB | RTX 3090 (24GB) |
| 13B | 7.8 GB | ~0.3 GB | ~16 GB | RTX 3090 (24GB) |
| 33B | 19.5 GB | ~0.6 GB | ~30 GB | A100 40GB |
| 65B/70B | 38 GB | ~1.0 GB | ~48 GB | A100 80GB |

### 8.4 Paged Optimizer 机制

当 GPU 显存不足时，paged optimizer 将优化器状态卸载到 CPU 内存：

```
GPU 显存不足 → 触发 page fault → 将 optimizer state page 移至 CPU RAM
→ 需要更新时 → 从 CPU RAM 加载回 GPU → 完成参数更新 → 释放 GPU 页
```

配置方式：
```python
training_args = TrainingArguments(
    optim="paged_adamw_8bit",  # 或 "paged_adamw_32bit"
    ...
)
```

## 9. 吞吐瓶颈

### 9.1 LoRA 吞吐分析

主要瓶颈：
1. **前向传播**：基础模型计算仍为主要开销，LoRA 分支增加约 1-3% 计算量
2. **反向传播**：梯度仅流经 LoRA 参数，但需要通过冻结层传播激活梯度
3. **通信开销**：多卡训练时 LoRA 参数的 all-reduce 开销极小

### 9.2 QLoRA 特有瓶颈

1. **反量化开销**：每次前向传播需将 NF4 → BF16，增加约 5-10% 计算时间
2. **内存带宽**：4-bit 权重的解压缩受限于内存带宽
3. **Paged optimizer 延迟**：CPU-GPU 数据传输引入额外延迟

### 9.3 优化策略

```
策略                    | 效果           | 代价
-----------------------|---------------|------------------
增大 batch size        | 提高 GPU 利用率 | 更多显存
Flash Attention 2      | 减少激活显存    | 需要兼容硬件
Gradient checkpointing | 减少激活显存    | 增加 ~33% 计算
torch.compile          | 算子融合加速    | 首次编译耗时
序列长度截断           | 减少计算量      | 可能损失信息
```

### 9.4 实测吞吐参考

| 配置 | 模型 | GPU | 吞吐 (tokens/s) |
|------|------|-----|-----------------|
| LoRA r=16, bs=4 | LLaMA-7B | A100 80GB | ~3500 |
| QLoRA r=16, bs=4 | LLaMA-7B | RTX 3090 | ~1200 |
| QLoRA r=16, bs=2 | LLaMA-13B | RTX 3090 | ~600 |
| QLoRA r=16, bs=1 | LLaMA-70B | A100 80GB | ~350 |

## 10. 分布式训练策略

### 10.1 LoRA 的分布式方案

由于 LoRA 可训练参数极少，分布式策略与全参数微调不同：

**数据并行（Data Parallel, DP）：**
- 最常用方案：每张卡加载完整模型 + LoRA
- 梯度同步开销极小（仅同步 LoRA 参数梯度）
- 适用于模型能放入单卡的场景

**FSDP/DeepSpeed ZeRO：**
- 当模型本身无法放入单卡时使用
- ZeRO Stage 3：分片模型参数 + LoRA 参数
- 注意：需正确配置 frozen parameter 的分片策略

### 10.2 QLoRA 多卡训练

```python
# DeepSpeed ZeRO-3 + QLoRA 配置
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},
        "offload_optimizer": {"device": "cpu"}
    },
    "bf16": {"enabled": true},
    "gradient_clipping": 0.3
}
```

### 10.3 注意事项

- QLoRA + FSDP 兼容性：需要 bitsandbytes >= 0.43.0
- 量化模型的分片：确保量化状态（absmax）随参数一起分片
- 通信精度：LoRA 梯度使用 BF16 通信即可，无需 FP32

## 11. Rollout 设计

### 11.1 LoRA 在 RLHF 中的 Rollout

在 RLHF/DPO 流程中，LoRA 模型需要生成 rollout：

```
Policy Model (Base + LoRA_policy) → 生成 response
Reference Model (Base + LoRA_ref 或 Base only) → 计算 KL penalty
Reward Model (Base + LoRA_rm) → 评分
```

### 11.2 权重合并加速推理

```python
# 合并 LoRA 权重以加速 rollout 生成
model = model.merge_and_unload()
# 此时模型等价于全参数模型，推理速度无损
```

### 11.3 多 LoRA 服务

使用 vLLM 的多 LoRA 支持：
```python
from vllm import LLM
llm = LLM(model="base_model", enable_lora=True, max_loras=4)
# 同时服务多个 LoRA adapter，共享基础模型 KV cache
```

### 11.4 Rollout 效率优化

- **批量生成**：将多个 prompt 打包为一个 batch
- **KV cache 复用**：相同 prefix 的 prompt 共享 KV cache
- **投机解码（Speculative Decoding）**：用小模型加速大模型生成

## 12. 评估指标

### 12.1 训练过程指标

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| train_loss | 训练损失 | 持续下降，最终 1.0-2.5 |
| eval_loss | 验证损失 | 与 train_loss 差距 < 0.3 |
| grad_norm | 梯度范数 | 0.1-10.0，无剧烈波动 |
| learning_rate | 当前学习率 | 按 schedule 变化 |

### 12.2 模型质量指标

| 指标 | 工具 | 说明 |
|------|------|------|
| Perplexity | 内置 | 语言模型困惑度 |
| MMLU | lm-eval-harness | 多任务知识评估 |
| HumanEval | 代码评估 | 代码生成能力 |
| MT-Bench | FastChat | 多轮对话质量 |
| AlpacaEval | 自动评估 | 指令遵循能力 |

### 12.3 LoRA 特有评估

- **Rank 消融实验**：对比不同 rank 的性能-效率曲线
- **Target module 消融**：评估不同层组合的效果
- **与全参数微调对比**：衡量 LoRA 的性能损失
- **合并后推理一致性**：确保 merge 前后输出一致

## 13. 常见失败模式

### 13.1 训练不收敛

**症状**：loss 不下降或剧烈震荡
**原因**：
- 学习率过高（LoRA 推荐 1e-4 ~ 5e-4，QLoRA 推荐 1e-4 ~ 2e-4）
- alpha/r 比值不当导致更新幅度异常
- 数据质量问题（标签噪声、格式错误）

### 13.2 过拟合

**症状**：train_loss 下降但 eval_loss 上升
**原因**：
- rank 过高导致参数过多
- 数据量不足（建议至少 1000 条高质量样本）
- 训练 epoch 过多

### 13.3 灾难性遗忘

**症状**：新任务性能提升但通用能力下降
**原因**：
- rank 过高 + 学习率过大
- 训练数据分布过于狭窄
- 缺少通用数据混合

### 13.4 QLoRA 特有问题

**量化误差累积**：
- NF4 量化引入的误差在深层网络中累积
- 解决：使用 BF16 compute dtype，确保 LoRA 分支精度

**Paged optimizer OOM**：
- CPU 内存不足导致 page 失败
- 解决：增加系统 swap 或减少 batch size

### 13.5 LoRA 合并异常

**症状**：合并后推理结果与训练时不一致
**原因**：
- alpha/r 缩放未正确应用
- 量化模型合并时精度损失
- 解决：先反量化再合并，或使用 `merge_and_unload()` API

## 14. Debug Checklist

```
□ 1. 数据检查
  □ 数据格式是否正确（instruction/input/output 或 messages）
  □ tokenization 后 labels 是否正确（-100 mask）
  □ special tokens 是否正确添加
  □ 数据长度分布是否合理

□ 2. 模型配置检查
  □ target_modules 是否覆盖关键层
  □ rank 和 alpha 设置是否合理（alpha = 2r 为常见起点）
  □ lora_dropout 是否适当（0.05-0.1）
  □ 可训练参数量是否符合预期（打印 model.print_trainable_parameters()）

□ 3. 训练配置检查
  □ 学习率是否在合理范围（1e-4 ~ 5e-4）
  □ warmup steps 是否足够（总步数的 3-10%）
  □ gradient_checkpointing 是否启用
  □ bf16/fp16 精度设置是否正确

□ 4. QLoRA 特有检查
  □ bnb_4bit_quant_type 是否为 "nf4"
  □ bnb_4bit_compute_dtype 是否为 bfloat16
  □ double_quant 是否启用
  □ 模型加载后是否正确冻结

□ 5. 运行时检查
  □ GPU 利用率是否正常（> 80%）
  □ 显存使用是否符合预期
  □ loss 曲线是否正常下降
  □ grad_norm 是否稳定

□ 6. 评估检查
  □ 生成结果是否合理（手动检查 10-20 条）
  □ 是否出现重复/退化输出
  □ 与基础模型对比是否有提升
  □ 合并后推理是否一致
```

## 15. 实验设计

### 15.1 Rank 消融实验

**目标**：确定最优 rank 值

| 实验组 | Rank | Alpha | 可训练参数 | 预期结果 |
|--------|------|-------|-----------|----------|
| A | 4 | 8 | 0.2% | 基线，可能欠拟合 |
| B | 8 | 16 | 0.4% | 简单任务足够 |
| C | 16 | 32 | 0.8% | 通用推荐 |
| D | 32 | 64 | 1.6% | 复杂任务 |
| E | 64 | 128 | 3.2% | 接近全参数 |
| F | 128 | 256 | 6.4% | 对比上限 |

**控制变量**：相同数据、相同训练步数、相同学习率
**评估**：下游任务准确率 + 训练时间 + 显存占用

### 15.2 Target Module 消融

| 实验组 | Target Modules | 说明 |
|--------|---------------|------|
| attn_only | q_proj, v_proj | 原始 LoRA 论文配置 |
| attn_full | q,k,v,o_proj | 完整 attention |
| mlp_only | gate, up, down_proj | 仅 MLP 层 |
| all_linear | attn + mlp | 所有线性层 |

### 15.3 QLoRA vs LoRA 对比实验

**固定条件**：相同模型、数据、rank、训练步数
**变量**：量化方式（FP16 vs NF4）
**指标**：最终 loss、下游任务分数、训练速度、显存占用

### 15.4 学习率搜索

推荐使用 log-uniform 搜索：[5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

## 16. 工程案例

### 16.1 案例：QLoRA 微调 LLaMA-3 8B 做中文对话

**背景**：在单张 RTX 4090 (24GB) 上微调 LLaMA-3-8B 用于中文客服场景

**配置**：
```python
# 模型加载
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)

# LoRA 配置
peft_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**数据**：5 万条中文客服对话，平均长度 512 tokens
**训练**：3 epochs，lr=2e-4，cosine schedule，batch_size=4，grad_accum=8
**结果**：
- 显存占用：18.5 GB
- 训练时间：6 小时
- MT-Bench 中文分数：从 5.2 提升到 7.1

### 16.2 案例：多 LoRA 服务架构

**场景**：同一基础模型服务多个客户的定制化需求

```
Base Model (共享，NF4/FP16)
├── LoRA_客户A (法律领域)
├── LoRA_客户B (医疗领域)
├── LoRA_客户C (金融领域)
└── LoRA_客户D (教育领域)
```

**实现**：使用 vLLM 的 multi-LoRA 功能
```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM("base_model", enable_lora=True, max_loras=4, max_lora_rank=64)

# 请求时指定 LoRA
output = llm.generate(
    "法律问题...",
    SamplingParams(temperature=0.7),
    lora_request=LoRARequest("legal_lora", 1, "path/to/legal_lora")
)
```

### 16.3 案例：LoRA 合并与量化部署

**流程**：
1. QLoRA 训练 → 保存 adapter
2. 加载 FP16 基础模型 + adapter → merge_and_unload()
3. 合并后模型 → GPTQ/AWQ 量化 → 部署

```python
# 合并
base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")

# 后续可用 AutoGPTQ 量化为 4-bit 部署
```

## 17. 习题 20 道

**基础概念（1-5）**

1. 解释 LoRA 中低秩分解的数学原理。为什么预训练模型的权重更新具有低秩特性？

2. 给定一个 hidden_size=4096 的 Transformer 层，计算 rank=16 时 LoRA 适配器的参数量，并与原始权重矩阵对比。

3. 解释 alpha 参数的作用。当 alpha=32, rank=16 时，实际的缩放因子是多少？如果将 rank 改为 32 但保持 alpha=32，缩放因子如何变化？

4. NF4 量化为什么比均匀量化（uniform quantization）更适合预训练权重？从信息论角度解释。

5. Double quantization 节省了多少显存？以 7B 模型为例进行具体计算。

**工程实践（6-10）**

6. 在 LLaMA 架构中，哪些层适合作为 LoRA 的 target modules？为什么通常建议同时包含 attention 和 MLP 层？

7. 设计一个实验来确定最优 rank。需要考虑哪些评估维度？

8. QLoRA 训练时，为什么 compute_dtype 建议使用 bfloat16 而非 float16？

9. 解释 paged optimizer 的工作原理。在什么场景下它会被触发？

10. 如何在不增加显存的情况下提高 QLoRA 的训练吞吐量？列出至少 3 种方法。

**进阶分析（11-15）**

11. LoRA 的 B 矩阵初始化为零、A 矩阵使用 Kaiming 初始化，这样设计的原因是什么？如果反过来会怎样？

12. 分析 LoRA rank 与模型容量的关系。是否存在一个 rank 阈值，超过后性能不再提升？

13. 在 RLHF 流程中使用 LoRA 时，policy model 和 reference model 如何高效共享基础模型？

14. 比较 LoRA、Adapter、Prefix-tuning 三种 PEFT 方法的优缺点。

15. QLoRA 论文中提到 NF4 + Double Quant + Paged Optimizer 三者缺一不可，解释为什么。

**故障排查（16-20）**

16. 训练过程中 loss 突然变为 NaN，可能的原因有哪些？如何排查？

17. LoRA 合并后模型输出与训练时不一致，列出可能的原因和解决方案。

18. QLoRA 训练速度比预期慢 50%，如何诊断和优化？

19. 使用 rank=64 训练后发现严重过拟合，除了降低 rank 外还有哪些解决方案？

20. 多卡 QLoRA 训练时出现显存不均衡，如何解决？

## 18. 标准答案

**1.** LoRA 基于"内在维度"假设：预训练模型在微调时，权重变化 ΔW 位于一个低维子空间中。Aghajanyan et al. (2020) 实验证明，即使将优化限制在很低的维度（如 d=100），模型仍能达到全参数微调 90% 的性能。因此 ΔW = BA，其中 B∈R^{d×r}, A∈R^{r×k}，r << min(d,k)。

**2.** 原始参数量：4096 × 4096 = 16,777,216。LoRA 参数量：16 × (4096 + 4096) = 131,072。压缩比：131,072 / 16,777,216 = 0.78%。若对 q,k,v,o 四个投影层都加 LoRA，总 LoRA 参数 = 131,072 × 4 = 524,288。

**3.** 缩放因子 = alpha/rank = 32/16 = 2。当 rank=32, alpha=32 时，缩放因子 = 32/32 = 1。缩放因子减半意味着每步更新幅度减半，可能需要相应调整学习率。

**4.** 预训练权重近似服从正态分布 N(0, σ²)。均匀量化对所有区间分配相同的码字，但正态分布的概率密度在均值附近最高。NF4 按等概率原则分配码字（每个量化区间包含相同概率质量），使得高密度区域获得更精细的表示，最小化期望量化误差。信息论上，这等价于最大化量化后的互信息 I(W; W_q)。

**5.** 7B 模型参数量约 7×10⁹。无 double quant：每 64 参数一个 FP32 constant = 32/64 = 0.5 bit/param，总额外显存 = 7×10⁹ × 0.5 / 8 = 437.5 MB。有 double quant：每参数额外约 0.127 bit，总额外显存 = 7×10⁹ × 0.127 / 8 ≈ 111 MB。节省约 326 MB。

**6.** LLaMA 架构中建议对所有线性层加 LoRA：q_proj, k_proj, v_proj, o_proj（attention）+ gate_proj, up_proj, down_proj（MLP）。原因：(1) MLP 层占模型参数的 2/3，仅对 attention 加 LoRA 会限制模型容量；(2) 实验表明 all-linear 配置在相同 rank 下性能最优；(3) 额外参数量增加有限但性能提升显著。

**7.** 实验设计：(1) 固定数据集和训练配置；(2) rank 取 [4,8,16,32,64,128]；(3) 每组训练相同步数；(4) 评估维度：下游任务准确率、训练时间、显存占用、推理速度；(5) 绘制 rank-performance 曲线找到拐点；(6) 考虑统计显著性，每组跑 3 次取均值。

**8.** BF16 的动态范围（exponent 8 bit）与 FP32 相同，不会出现 FP16 的溢出问题（FP16 最大值约 65504）。在反量化计算中，中间结果可能超出 FP16 范围导致 Inf/NaN。BF16 虽然精度略低（mantissa 7 bit vs 10 bit），但训练稳定性更好。

**9.** Paged optimizer 利用 NVIDIA 统一内存（Unified Memory）机制。当 GPU 显存不足时，CUDA 驱动自动将不活跃的内存页迁移到 CPU RAM。在 optimizer step 时，按需将对应参数的 optimizer state 从 CPU 加载回 GPU。触发条件：GPU 显存使用接近上限时的 page fault。代价是 PCIe 带宽成为瓶颈。

**10.** (1) 使用 Flash Attention 2 减少激活显存，允许更大 batch size；(2) torch.compile 进行算子融合；(3) 增大 gradient_accumulation_steps 减少通信频率；(4) 使用更长序列 + packing 提高 token 利用率；(5) 关闭不必要的 logging 和 evaluation。

**11.** B=0 确保训练开始时 ΔW=BA=0，模型从预训练状态开始，不引入随机扰动。A 使用 Kaiming 初始化确保前向传播方差稳定。如果反过来（A=0, B=Kaiming），效果相同（乘积仍为零）。但如果两者都随机初始化，训练开始时模型就偏离预训练状态，可能导致初始 loss 剧增和训练不稳定。

**12.** 存在有效 rank 上限。当 rank 超过权重更新的真实内在维度时，额外容量不带来性能提升反而增加过拟合风险。实验中通常 rank=64-128 后性能趋于饱和。具体阈值取决于任务复杂度和数据量。可通过 SVD 分析全参数微调的 ΔW 来估计内在维度。

**13.** 高效方案：基础模型权重共享，policy 和 reference 各自有独立的 LoRA adapter。前向传播时切换 adapter 即可。具体实现：(1) 加载一份基础模型；(2) policy LoRA 可训练；(3) reference LoRA 冻结（或直接用无 LoRA 的基础模型作为 reference）。vLLM 的 multi-LoRA 支持可同时服务两个 adapter。

**14.** LoRA：优点是无推理延迟（可合并）、参数高效、实现简单；缺点是对 rank 敏感。Adapter：优点是模块化好；缺点是增加推理延迟（额外层）。Prefix-tuning：优点是不修改模型结构；缺点是占用序列长度、难以优化、性能通常不如 LoRA。综合来看 LoRA 是当前最优选择。

**15.** NF4 将模型压缩到 4-bit 使其能放入单卡；Double Quant 进一步节省量化常数的显存（~300MB for 65B）使得 LoRA + 激活值有足够空间；Paged Optimizer 处理训练过程中的显存峰值（optimizer step 时的临时显存需求）。三者协同才能在 48GB 显存上训练 65B 模型。

**16.** 可能原因：(1) 学习率过高导致梯度爆炸；(2) FP16 溢出（应使用 BF16）；(3) 数据中存在异常值；(4) gradient clipping 未启用。排查步骤：检查 grad_norm 历史、检查数据 batch、尝试降低 lr、启用 max_grad_norm=1.0、切换到 BF16。

**17.** 原因：(1) 合并时 scaling factor 计算错误（alpha/r 未正确应用）；(2) 从量化模型合并导致精度损失；(3) tokenizer 配置不一致。解决：(1) 确认 PEFT 版本正确处理 scaling；(2) 先将基础模型加载为 FP16 再合并；(3) 合并后对比几条样本的 logits。

**18.** 诊断：(1) 检查 GPU 利用率（nvidia-smi）；(2) 检查是否触发 paged optimizer（CPU-GPU 传输）；(3) 检查数据加载是否为瓶颈（num_workers）；(4) 检查是否有不必要的同步操作。优化：启用 Flash Attention、增大 batch size、使用 torch.compile、确保 compute_dtype=bf16。

**19.** 解决方案：(1) 增加 lora_dropout（0.1-0.2）；(2) 减少训练 epochs；(3) 增加训练数据量；(4) 使用 early stopping；(5) 添加 weight decay；(6) 混合通用数据防止过拟合到特定分布；(7) 减少 target_modules 数量。

**20.** 原因：device_map="auto" 可能不均匀分配层。解决：(1) 手动指定 device_map 确保均匀分配；(2) 使用 FSDP/DeepSpeed ZeRO-3 自动分片；(3) 确保量化模型的每层大小一致；(4) 使用 accelerate 的 balanced 策略。

## 19. 面试题 20 道

1. 请解释 LoRA 的核心思想，为什么它能在极少参数下达到接近全参数微调的效果？

2. LoRA 中 rank 的选择有什么经验法则？rank 过高或过低分别会导致什么问题？

3. alpha 参数和 rank 的关系是什么？为什么通常设置 alpha = 2 × rank？

4. QLoRA 使用 NF4 而非 INT4 量化的原因是什么？从数学角度解释。

5. Double quantization 的具体实现是什么？它节省了多少显存？

6. 在生产环境中，如何高效服务多个 LoRA adapter？

7. LoRA 权重合并（merge）的数学过程是什么？合并后有什么优势和限制？

8. 比较 LoRA 和全参数微调在以下场景的适用性：(a) 数据量 100 条 (b) 数据量 100 万条 (c) 需要学习全新语言

9. QLoRA 训练时梯度如何通过量化层传播？是否需要 STE？

10. 如何判断 LoRA 训练是否成功？列出关键的评估指标和方法。

11. 在 RLHF 流程中使用 LoRA 有什么特殊考虑？如何处理 reference model？

12. LoRA 的 dropout 机制与标准 dropout 有什么区别？它在什么情况下有帮助？

13. 解释为什么 LoRA 不会增加推理延迟，而 Adapter 方法会。

14. 如果训练后发现模型在某些能力上退化（灾难性遗忘），有哪些缓解策略？

15. QLoRA 的 paged optimizer 在什么情况下会显著降低训练速度？如何避免？

16. 设计一个实验来验证 LoRA 是否适合你的特定任务（vs 全参数微调）。

17. 在分布式训练中，LoRA 参数的同步策略与全参数有什么不同？

18. 如何将 QLoRA 训练的模型部署到生产环境？完整流程是什么？

19. LoRA 能否用于预训练阶段？为什么通常只用于微调？

20. 最新的 LoRA 变体（如 DoRA、rsLoRA、LoRA+）解决了什么问题？

## 20. 复习卡片 30 张

**卡片 1**
Q: LoRA 的全称是什么？核心思想一句话概括。
A: Low-Rank Adaptation。通过将权重更新分解为两个低秩矩阵的乘积，实现参数高效微调。

**卡片 2**
Q: LoRA 的前向传播公式是什么？
A: h = W₀x + (α/r)BAx，其中 B∈R^{d×r}, A∈R^{r×k}。

**卡片 3**
Q: LoRA 中 B 和 A 矩阵分别如何初始化？为什么？
A: B 初始化为零，A 使用 Kaiming 初始化。确保训练开始时 ΔW=0，模型从预训练状态出发。

**卡片 4**
Q: alpha/r 缩放因子的作用是什么？
A: 使得不同 rank 设置下，LoRA 更新的幅度保持一致，方便超参数迁移。

**卡片 5**
Q: 对于 d=4096, r=16 的 LoRA，参数压缩比是多少？
A: r(d+k)/(dk) = 16×8192/(4096²) = 0.78%。

**卡片 6**
Q: QLoRA 的三个关键技术是什么？
A: NF4 量化、Double Quantization、Paged Optimizer。

**卡片 7**
Q: NF4 量化的核心假设是什么？
A: 预训练权重近似服从正态分布，因此按等概率原则分配量化码字可最小化量化误差。

**卡片 8**
Q: Double Quantization 对什么进行二次量化？
A: 对第一次量化产生的 FP32 scaling constants 进行 FP8 量化。

**卡片 9**
Q: QLoRA 7B 模型大约需要多少显存？
A: 约 10GB（NF4 模型 4.2GB + LoRA + 优化器 + 激活值）。

**卡片 10**
Q: Paged Optimizer 的触发条件是什么？
A: GPU 显存不足时，通过 CUDA 统一内存的 page fault 机制将 optimizer state 迁移到 CPU。

**卡片 11**
Q: LoRA 推荐的学习率范围是多少？
A: 1e-4 到 5e-4，QLoRA 通常使用 1e-4 到 2e-4。

**卡片 12**
Q: 为什么 QLoRA 推荐使用 BF16 而非 FP16 作为 compute dtype？
A: BF16 动态范围与 FP32 相同（8-bit exponent），避免 FP16 的溢出问题。

**卡片 13**
Q: LoRA 相比 Adapter 方法的最大优势是什么？
A: 无推理延迟——LoRA 权重可合并回原模型，而 Adapter 增加额外计算层。

**卡片 14**
Q: 如何选择 LoRA 的 target modules？
A: 推荐对所有线性层（attention + MLP）都加 LoRA，实验表明 all-linear 效果最优。

**卡片 15**
Q: LoRA rank 选择的经验法则？
A: 简单任务 4-8，通用微调 16-32，复杂任务 64-128。通过消融实验确定最优值。

**卡片 16**
Q: LoRA 训练过拟合的信号是什么？
A: train_loss 持续下降但 eval_loss 上升，生成结果开始重复训练数据中的模式。

**卡片 17**
Q: LoRA 合并的数学操作是什么？
A: W_merged = W₀ + (α/r) × B × A，合并后模型结构与原模型完全相同。

**卡片 18**
Q: 多 LoRA 服务的核心优势是什么？
A: 多个客户共享一份基础模型权重和 KV cache，每个 LoRA adapter 仅占几十 MB。

**卡片 19**
Q: QLoRA 训练时梯度是否流经量化的基础模型？
A: 不需要。基础模型被冻结，梯度仅通过 LoRA 分支（BA）计算和更新。

**卡片 20**
Q: gradient checkpointing 在 LoRA 训练中的作用？
A: 用计算换显存——不保存中间激活值，反向传播时重新计算，节省约 60% 激活显存。

**卡片 21**
Q: LoRA dropout 应用在哪里？
A: 应用在 LoRA 的输出上（BA 的结果），训练时随机置零部分输出以防止过拟合。

**卡片 22**
Q: 如何验证 LoRA 合并是否正确？
A: 对比合并前后相同输入的 logits 输出，差异应为数值精度级别（< 1e-5）。

**卡片 23**
Q: LoRA 在 RLHF 中的典型用法？
A: Policy model = Base + LoRA_trainable，Reference model = Base（无 LoRA），共享基础模型权重。

**卡片 24**
Q: QLoRA 的量化块大小（block size）通常是多少？
A: 64，即每 64 个参数共享一个 scaling constant。

**卡片 25**
Q: LoRA 不适用的场景有哪些？
A: (1) 需要学习全新语言/模态；(2) 预训练数据与目标分布差异极大；(3) 需要修改模型架构。

**卡片 26**
Q: rsLoRA 解决了什么问题？
A: 标准 LoRA 的 α/r 缩放在高 rank 时不稳定，rsLoRA 使用 α/√r 使不同 rank 的训练动态更一致。

**卡片 27**
Q: 如何估算 LoRA 训练的总显存需求？
A: 模型权重 + LoRA 参数×2（梯度）+ LoRA 参数×8（AdamW 状态）+ 激活值（取决于 batch/seq_len）。

**卡片 28**
Q: LoRA 训练中 grad_norm 的健康范围？
A: 通常 0.1-10.0，无剧烈波动。持续 > 100 说明学习率过高或数据异常。

**卡片 29**
Q: QLoRA 与 GPTQ/AWQ 量化的区别？
A: QLoRA 是训练时量化（保留梯度路径），GPTQ/AWQ 是推理时量化（后训练压缩，不可训练）。

**卡片 30**
Q: LoRA 的可训练参数比例通常是多少？
A: 0.1% - 3%，取决于 rank 和 target modules 数量。7B 模型 rank=16 all-linear 约 0.8%。
