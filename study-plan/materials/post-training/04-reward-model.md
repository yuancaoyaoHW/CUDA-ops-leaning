# 04 - Reward Model（奖励模型）

## 1. 学习目标

- 理解奖励模型（Reward Model, RM）在 RLHF 流程中的核心作用
- 掌握 Bradley-Terry 模型的数学推导与训练目标
- 了解奖励模型的数据标注流程与质量控制
- 能够独立完成奖励模型的训练、评估与部署
- 理解奖励模型的局限性及其对下游 RL 训练的影响

## 2. 方法动机

人类偏好难以用规则或启发式方法精确刻画。奖励模型通过学习人类对比偏好（pairwise comparison），将主观判断转化为标量信号，为强化学习提供可微分的优化目标。相比直接用人类反馈做在线训练，RM 可以：
- 大幅降低标注成本（离线训练后可无限次推理）
- 提供平滑的梯度信号
- 支持批量化的 RL 训练流程

## 3. 核心术语表

| 术语 | 英文全称 | 含义 |
|------|----------|------|
| 奖励模型 | Reward Model (RM) | 将 (prompt, response) 映射为标量分数的模型 |
| Bradley-Terry 模型 | Bradley-Terry Model | 基于配对比较的概率排序模型 |
| 偏好数据 | Preference Data | 人类标注的 chosen/rejected 对 |
| 标注者间一致性 | Inter-Annotator Agreement (IAA) | 多个标注者对同一样本判断的一致程度 |
| 奖励头 | Reward Head | 模型最后一层输出标量分数的线性层 |
| 边际 | Margin | chosen 与 rejected 分数之差 |
| 校准 | Calibration | 模型输出分数与真实偏好概率的对齐程度 |
| 过拟合 | Overfitting | 模型在训练集上表现好但泛化差 |

## 4. 数学定义

### 4.1 Bradley-Terry 模型

给定 prompt $x$，两个回复 $y_w$（chosen）和 $y_l$（rejected），人类偏好概率建模为：

$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $r_\theta(x, y)$ 为奖励模型输出的标量分数，$\sigma$ 为 sigmoid 函数。

### 4.2 Plackett-Luce 扩展

对于 K 个回复的排序数据：

$$P(\pi | x) = \prod_{k=1}^{K} \frac{\exp(r_\theta(x, y_{\pi(k)}))}{\sum_{j=k}^{K} \exp(r_\theta(x, y_{\pi(j)}))}$$

