# 03 - 全参数微调与序列打包（Full Fine-Tuning & Sequence Packing）

---

## 1. 学习目标

- 理解全参数微调（Full Fine-Tuning）与参数高效微调的本质区别
- 掌握序列打包（Sequence Packing）的动机、实现方式与注意事项
- 能够独立配置一个支持 Packing 的全参数 SFT 训练流程
- 理解 Packing 对 Attention Mask、Loss 计算、数据效率的影响
- 掌握显存估算与吞吐优化的工程实践

---

## 2. 方法动机

### 2.1 为什么需要全参数微调

LoRA/QLoRA 虽然显存友好，但在以下场景中全参数微调仍不可替代：

- **领域迁移幅度大**：预训练分布与目标分布差异显著（如代码模型转医疗）
- **模型容量充足**：有足够 GPU 资源时，全参数更新能获得更好的 loss 收敛
- **知识注入密集**：需要大量新知识写入权重（而非仅调整输出风格）
- **生产部署无适配器开销**：全参数微调后无需 adapter merge 步骤

### 2.2 为什么需要序列打包

标准 SFT 训练中，每条样本 pad 到 max_seq_len，导致：

- 短样本占比高时，大量计算浪费在 padding token 上
- GPU 利用率（MFU, Model FLOPs Utilization）显著下降
- 训练吞吐（tokens/sec）远低于理论峰值

Packing 将多条短样本拼接到一个固定长度序列中，消除 padding 浪费。

---

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| 全参数微调 | Full Fine-Tuning | 更新模型所有参数的训练方式 |
| 序列打包 | Sequence Packing | 将多条样本拼接为一个固定长度序列的数据处理技术 |
| 注意力掩码 | Attention Mask | 控制 token 间可见性的二值矩阵 |
| 文档边界 | Document Boundary | Packing 中不同样本的分界位置 |
| 因果语言模型 | Causal Language Model (CLM) | 仅关注左侧上下文的自回归模型 |
| 模型浮点利用率 | Model FLOPs Utilization (MFU) | 实际计算量占硬件峰值的比例 |
| 梯度累积 | Gradient Accumulation | 多个 micro-batch 梯度求和后再更新参数 |
| 混合精度训练 | Mixed Precision Training | 同时使用 FP16/BF16 和 FP32 的训练方式 |
| 学习率预热 | Learning Rate Warmup | 训练初期逐步增大学习率的策略 |
| 余弦退火 | Cosine Annealing | 学习率按余弦函数衰减的调度策略 |

---

## 4. 数学定义

### 4.1 标准 CLM Loss

给定序列 $x = (x_1, x_2, \ldots, x_T)$，自回归语言模型的负对数似然损失：

$$\mathcal{L}_{\text{CLM}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})$$

### 4.2 SFT Loss（仅计算 response 部分）

对于 instruction-response 对，设 response 起始位置为 $s$：

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{T-s} \sum_{t=s}^{T} \log p_\theta(x_t | x_{<t})$$

### 4.3 Packing 下的 Loss

设一个 packed 序列包含 $K$ 条样本，第 $k$ 条样本的 response token 集合为 $\mathcal{R}_k$：

$$\mathcal{L}_{\text{pack}} = -\frac{1}{\sum_k |\mathcal{R}_k|} \sum_{k=1}^{K} \sum_{t \in \mathcal{R}_k} \log p_\theta(x_t | x_{<t}^{(k)})$$

其中 $x_{<t}^{(k)}$ 表示第 $k$ 条样本内 $t$ 之前的 token（跨文档不可见）。

### 4.4 Packing Attention Mask

Block-diagonal attention mask $M \in \{0,1\}^{L \times L}$：

$$M_{ij} = \begin{cases} 1 & \text{if } i \geq j \text{ and } \text{doc}(i) = \text{doc}(j) \\ 0 & \text{otherwise} \end{cases}$$

---

## 5. Loss 推导

### 5.1 朴素 Padding 方式的 Loss

