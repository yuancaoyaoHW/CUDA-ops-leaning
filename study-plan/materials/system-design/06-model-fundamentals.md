# 模型基础兜底题

## 学习目标

1. 能解释 Transformer 中 attention, FFN, LayerNorm/RMSNorm, positional encoding 的作用。
2. 能回答 AdamW, LR schedule, gradient clipping, loss spike 等训练稳定性问题。
3. 能说明 Scaling Law, Chinchilla compute-optimal 和 emergent abilities 的边界。
4. 能比较 BPE, SentencePiece, vocab size 对训练和推理的影响。
5. 能讲清 long context 中 RoPE scaling, ALiBi, sliding window 的差异。
6. 能解释 MoE routing, load balancing loss, capacity factor 和推理负载不均。

## 前置知识

- 神经网络基础: embedding, linear, activation, residual。
- 概率基础: softmax, cross entropy, perplexity。
- 训练基础: mini-batch, optimizer, learning rate, gradient。
- 推理基础: autoregressive decoding, KV cache, greedy/sample。


## 核心内容

Decoder-only LLM 的主线是: token -> embedding -> 多层 Transformer block -> LM head -> next-token logits。每层通过 attention 混合上下文, 通过 FFN 做非线性变换, 通过 residual 和 norm 保持深层训练稳定。

面试兜底原则:

1. 先给公式或结构, 再解释工程含义。
2. 把模型概念连接到训练稳定性或推理成本。
3. 对 tradeoff 说边界, 不把经验结论说成绝对规律。


## 完整的问答/题目
### Q1: Decoder-only Transformer 一层里发生了什么?
**考察点:** block 主路径, attention, FFN, residual, norm。
**参考答案:**

现代 LLM 常用 Pre-LN:

```text
x = x + SelfAttention(Norm(x))
x = x + FFN(Norm(x))
```

SelfAttention 让当前位置读取历史 token 信息。FFN 对每个 token 的 hidden state 独立做非线性变换。Residual connection 保留梯度高速通道, Norm 控制激活尺度, 提高深层训练稳定性。

推理分两段:

- Prefill: 一次处理 prompt, 生成所有位置的 KV cache。
- Decode: 每步只处理新 token, 但每层 attention 要读取历史 KV。

这就是为什么 prefill 更偏 compute-heavy, decode 更偏 memory-bandwidth-heavy。
**追问方向:** Pre-LN 和 Post-LN 有什么差异? 为什么现代 LLM 常用 RMSNorm?

### Q2: Attention 的 Q, K, V 是什么? 为什么除以 sqrt(d)?
**考察点:** attention 公式, softmax 稳定性, GQA/MQA。
**参考答案:**

公式:

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_head) + mask) V
```

Q 表示当前位置想查什么, K 表示每个位置可被什么匹配, V 表示匹配后读取的信息。`QK^T` 是相关性分数。

除以 `sqrt(d_head)` 是为了控制 dot product 方差。head_dim 越大, 未缩放分数方差越大, softmax 越容易饱和, 梯度和训练稳定性越差。

MHA 每个 query head 有独立 K/V。GQA 让多个 query heads 共享一组 K/V, MQA 只有一组 K/V。GQA/MQA 主要收益是降低 KV cache 容量和 decode 读带宽。
**追问方向:** Causal mask 的作用是什么? GQA 会不会影响质量?

### Q3: FFN 的作用是什么? SwiGLU 和 GELU FFN 有什么区别?
**考察点:** FFN 结构, activation, 参数量。
**参考答案:**

Attention 负责 token 间信息交互, FFN 负责每个 token 内部的特征变换。普通 FFN:

```text
FFN(x) = W2 activation(W1 x)
```

SwiGLU 是 gated FFN:

```text
SwiGLU(x) = W_down(SiLU(W_gate x) * W_up x)
```

Gating 提高表达能力, 现代 LLaMA 类模型常用。代价是多一路 projection, GEMM 形状和参数量不同。推理中 FFN 通常是大 GEMM, 更适合 Tensor Core 和 epilogue fusion。
**追问方向:** 为什么 FFN 常占参数大头? SwiGLU 为什么常见于大模型?

### Q4: LayerNorm 和 RMSNorm 有什么区别?
**考察点:** normalization, reduction, 数值稳定性。
**参考答案:**

LayerNorm:

```text
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm:

```text
y = x / sqrt(mean(x^2) + eps) * weight
```

RMSNorm 不减均值, 少一次 mean 相关计算和 beta 参数, 实现更简单。大模型里它通常足够稳定, 推理中也更适合和 residual add, quantization fuse。

Norm 的 eps 很关键, 防止除以过小数导致数值不稳定。Norm kernel 通常是 memory-bound, 因为计算少但要读写整行 hidden。
**追问方向:** Norm 为什么放在 attention/FFN 之前更稳定? eps 过大有什么问题?

### Q5: Positional encoding 为什么必要? RoPE, ALiBi, absolute position 怎么比较?
**考察点:** 位置信息, extrapolation, long context。
**参考答案:**

Self-attention 本身对顺序不敏感, 所以必须注入位置信息。

- Learned absolute position: 每个位置一个可学习向量, 简单, 但长度外推差。
- Sinusoidal: 固定正弦/余弦, 有一定外推能力。
- RoPE: 对 Q/K 按位置做旋转, attention score 中包含相对位置信息。现代 LLM 常见。
- ALiBi: 在 attention score 加随距离增加的负 bias, 鼓励关注近处, 外推较自然。

RoPE scaling 用于扩展上下文长度, 常见 linear scaling, NTK-aware scaling, YaRN。目标是让更长位置的旋转频率不要过度偏离训练分布。
**追问方向:** RoPE 为什么作用在 Q/K 而不是 V? RoPE scaling 是否影响 KV cache 复用?

### Q6: AdamW 和 Adam 的区别是什么?
**考察点:** optimizer, weight decay, optimizer state。
**参考答案:**

Adam 用一阶和二阶动量自适应更新:

```text
m_t = beta1*m_{t-1} + (1-beta1)*g_t
v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

AdamW 的关键是 decoupled weight decay:

```text
theta = theta - lr * adam_update - lr * weight_decay * theta
```

传统 Adam 中 L2 正则会被自适应缩放影响, AdamW 把参数衰减独立出来, 行为更可控。大模型常用 AdamW, 但 optimizer state 很大, m/v 通常带来显著显存或 CPU 内存压力。
**追问方向:** beta1/beta2 影响什么? 为什么 norm 和 bias 有时不做 weight decay?

### Q7: LR schedule, warmup, gradient clipping 各解决什么问题?
**考察点:** 学习率, 训练稳定性, 梯度爆炸。
**参考答案:**

Warmup 解决训练早期不稳定。参数和 Adam 动量统计还没稳定时, 直接用 peak LR 容易 loss spike 或 diverge。

常见 schedule:

```text
warmup: lr 从 0 线性升到 peak
decay: cosine 或 linear 降到较小值
```

Gradient clipping 限制梯度范数:

```text
if ||g|| > threshold:
    g = g * threshold / ||g||
```

它不能修复坏数据或错误 mask, 但可以防止异常 batch 造成过大参数更新。LR, batch size, clipping threshold, AdamW beta 和 loss scale 是耦合的, 不能孤立调。
**追问方向:** warmup 太短会怎样? clipping 太小有什么副作用?

### Q8: 训练中 loss spike 如何排查?
**考察点:** 数据, 超参, 数值, 分布式训练。
**参考答案:**

按四层排查:

1. 数据: 是否有异常长样本, 空样本, tokenizer 错误, 数据源切换, sampling weight 改动。
2. 超参: LR 是否在 peak, warmup 是否太短, clipping 是否失效, AdamW beta/eps/weight_decay 是否改动。
3. 数值: FP16 loss scale, BF16 溢出, NaN/Inf gradient, fused kernel 或 mask bug。
4. 分布式: resume 后 optimizer state 是否完整, global batch 是否变化, DP/TP/PP 通信是否异常。

如果 spike 后快速恢复, 可能是异常 batch 或短期 LR 问题。如果长期不恢复, 要考虑回滚 checkpoint, 降 LR, 修数据或恢复 optimizer state。
**追问方向:** 如何定位是哪条数据导致 spike? Loss spike 一定导致 eval 退化吗?

### Q9: Scaling Law 和 Chinchilla compute-optimal 讲什么?
**考察点:** 参数量, token 数, compute budget。
**参考答案:**

Scaling Law 描述 loss 随参数量 N, 训练 token 数 D, compute C 增加而下降。Chinchilla 的重要结论是: 许多模型在固定 compute 下参数偏多, 训练 token 偏少。更 compute-optimal 的做法是同时增加参数量和训练 token。

简化经验:

```text
固定 compute 下, 不应只追求更大参数;
训练 token 数也要足够, 常见经验是 tokens/param 在几十量级。
```

对工程的影响:

- 参数越大, 推理权重显存和每 token compute 越高。
- token 训练更多可能提升质量, 但不增加推理参数量。
- compute-optimal 不等于 serving-optimal, 部署还要考虑 latency, cost, batch capacity。
**追问方向:** 为什么小模型多训 token 可能比大模型欠训练更好? Serving-optimal 如何定义?

### Q10: Emergent abilities 是什么? 如何谨慎回答?
**考察点:** 涌现能力, benchmark 解释, eval。
**参考答案:**

Emergent abilities 指某些能力随模型规模或训练 compute 增加后突然表现明显, 例如 few-shot reasoning, instruction following, multi-step arithmetic。

谨慎点:

- 有些涌现来自指标离散化。loss 平滑下降, accuracy 过阈值后看起来像突然跃迁。
- Prompt 格式, tokenization, decoding 都会影响 benchmark。
- 参数到某个规模不保证自动获得某能力。

工程上应说: scaling 提高表达和泛化能力, 但目标能力必须用 eval set 验证, 不能依赖 "规模到了自然会有"。
**追问方向:** loss 平滑下降为什么 accuracy 可能突然上升? 如何设计能力 eval?

### Q11: BPE 和 SentencePiece 的基本思想是什么?
**考察点:** tokenizer, subword, vocab 训练。
**参考答案:**

BPE 从基础符号开始, 反复合并语料中最高频的相邻 pair, 直到达到 vocab size。它把常见片段变成一个 token, 把罕见词拆成子词。

SentencePiece 是 tokenizer 工具和建模方式, 常见 BPE 或 Unigram LM。它把文本作为原始字符流处理, 不强依赖空格预分词, 对中文/日文等语言更友好。常用特殊符号表示空格, 让 detokenization 可逆。

Tokenizer 必须和训练完全一致。tokenizer 版本, special tokens, chat template 不一致, 可能不报错但输出质量严重退化。
**追问方向:** Byte-level BPE 为什么少 unknown token? SentencePiece 为什么适合多语言?

### Q12: Vocab size 如何影响训练和推理?
**考察点:** vocab tradeoff, embedding, LM head, 序列长度。
**参考答案:**

大 vocab:

- 优点: 常见词更短, 同一文本 token 数少, context window 容纳更多语义。
- 缺点: embedding/LM head 参数更大, logits softmax 更贵, 低频 token 学习不足。

小 vocab:

- 优点: 词表参数少, 稀有字符可组合, 覆盖更均匀。
- 缺点: 序列更长, attention 和 KV cache 成本增加。

中文, 代码, URL, emoji, 多语言都会影响 vocab 选择。代码需要保留缩进和符号模式, 多语言要避免高资源语言占满词表。
**追问方向:** Token 数变少是否一定更好? tokenizer 变了为什么模型权重不能直接复用?

### Q13: RoPE scaling, ALiBi, sliding window 分别解决什么问题?
**考察点:** long context 方法比较, 位置外推, 复杂度。
**参考答案:**

RoPE scaling 主要解决 RoPE 模型扩展到更长位置时的分布偏移, 通过调整位置到旋转频率的映射保留长位置可用性。风险是短上下文质量或位置分辨率下降。

ALiBi 在 attention score 上加随距离增加的负 bias, 鼓励关注近处, 对长度外推较自然, 但通常需要训练时采用。

Sliding window attention 让每个 token 只关注最近 W 个 token, 可加 sink/global tokens。它直接降低 attention 计算和 KV 访问成本, 但窗口外信息不能直接访问。

总结: RoPE scaling/ALiBi 主要处理位置外推, sliding window 主要处理复杂度和 KV 成本。
**追问方向:** Sliding window 是否减少 KV cache? Sink token 的作用是什么?

### Q14: Long context 下 prefill 和 decode 成本如何变化?
**考察点:** O(N^2), O(N), KV cache, FlashAttention 边界。
**参考答案:**

Dense attention prefill 复杂度约:

```text
O(N^2 * d)
```

FlashAttention 能减少 HBM IO, 避免 materialize attention matrix, 但不改变 dense attention 的理论计算量。

Decode 每步新 token 对历史 N 个 KV 做 attention:

```text
single-step decode attention: O(N * d)
KV cache memory: O(N * layers * kv_heads * head_dim)
```

因此长上下文有两个问题: TTFT 高, TPOT 变慢且 batch capacity 降低。Prefill 优化靠 FlashAttention, chunked prefill, prefix cache。Decode 优化靠 GQA/MQA, paged KV, KV quantization, sliding window, speculation。
**追问方向:** Decode 为什么通常 memory-bound? FlashAttention 为什么不能解决 KV 容量?

### Q15: MoE routing, load balancing loss, capacity factor 是什么?
**考察点:** MoE 基础, routing, balance, inference load。
**参考答案:**

MoE 把 FFN 换成多个 experts。Router 为每个 token 选择 top-k experts:

```text
router_probs = softmax(x W_router)
experts = top_k(router_probs)
output = sum router_prob_i * expert_i(x)
```

如果不约束, router 可能把 token 发给少数 experts, 导致过载。Load balancing loss 鼓励均匀:

```text
balance_loss = num_experts * sum_i(f_i * p_i)
```

`f_i` 是实际分给 expert i 的 token 比例, `p_i` 是平均 router probability。

Capacity factor 控制每个 expert 最大容量:

```text
capacity = ceil(tokens * top_k / num_experts * capacity_factor)
```

训练中超容量 token 可能 drop/reroute。推理中通常不希望 drop, 因为会影响输出质量, 所以更依赖 EP, hot expert replica, routing-aware batching。
**追问方向:** top-1/top-2 routing 差异是什么? MoE 推理为什么容易 p99 变差?


## 追问方向与深入点

1. Transformer: Pre-LN 稳定性, GQA 质量影响, FFN compute 占比, Norm kernel memory-bound。
2. 训练: LR warmup, AdamW state, clipping 阈值, loss spike 的数据和数值排查。
3. Scaling: Chinchilla 是训练 compute-optimal, 不是 serving-optimal; emergent ability 必须 eval。
4. Tokenization: vocab size 与 token 数, LM head 成本, 中文/代码/多语言 tradeoff。
5. Long context: RoPE scaling 是否继续训练, sliding window 对 RAG 全局依赖的风险。
6. MoE: balance loss 和 capacity factor 只解决训练均衡, 推理还要处理通信和 straggler。


## 评分标准

| 等级 | 表现 |
|------|------|
| A | 能写公式, 讲机制, 连接训练稳定性和推理成本, 能说明 tradeoff。 |
| B | 概念基本正确, 但 long context, MoE 或 optimizer 的工程影响不够深入。 |
| C | 能背术语, 但无法说明为什么需要该机制或影响什么指标。 |
| D | 混淆 attention/FFN/norm/tokenizer, 或把训练方法和推理优化混为一谈。 |

面试重点: decoder block 主路径, attention/KV cache 公式, AdamW/LR/clipping 职责, Scaling Law 对资源规划的含义, tokenizer 与模型权重强绑定, MoE balance 和 capacity 的边界。


## 复习卡片 15 张

1. Decoder block: `x=x+Attn(Norm(x)); x=x+FFN(Norm(x))`。
2. Q/K/V: Q 查什么, K 如何匹配, V 读什么。
3. 除以 sqrt(d): 控制 attention logits 方差, 防止 softmax 饱和。
4. GQA: 多个 Q heads 共享 K/V, 降低 KV cache。
5. FFN: 对每个 token 独立做非线性特征变换。
6. SwiGLU: `SiLU(gate) * up` 的 gated FFN。
7. RMSNorm: 不减均值, 用均方根缩放。
8. RoPE: 旋转 Q/K 注入相对位置信息。
9. AdamW: weight decay 与 Adam update 解耦。
10. Warmup: 避免训练早期 LR 过大导致发散。
11. Clipping: 限制全局梯度范数, 缓解异常 batch。
12. Chinchilla: 固定 compute 下参数和训练 token 要均衡。
13. BPE: 反复合并最高频相邻 pair 形成子词。
14. Vocab tradeoff: 大 vocab 序列短但 LM head 贵, 小 vocab 相反。
15. MoE capacity factor: expert 容量倍率, 权衡 drop 风险和计算浪费。

## 参考链接索引

本文出现的项目、论文、技术报告和博客链接集中维护在 [09-reference-links.md](./09-reference-links.md)。
