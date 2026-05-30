# Day 14：第 2 周复习 + 周检 + 阶段检 #1

## 学习目标
- 闭卷验证第 2 周所有核心内容
- 模拟面试检验表达能力
- 第一次阶段检（评估前 2 周整体水平）

---

## 上午（2h）- 闭卷手写

### 限时 50 分钟完成

**1. FlashAttention Rescaling 公式（10min）**
- 写出 m_new, l_new, O_new 的更新公式
- 解释为什么这保证了数值正确性

**2. vLLM Scheduler 流程图（10min）**
- 画出 waiting/running/swapped 三个队列
- 标注 prefill/decode/preempt/swap 的转换条件

**3. Speculative Decoding 算法（10min）**
- 写出 draft → verify → accept/reject 的完整步骤
- 写出 acceptance 条件: r < min(1, p(x)/q(x))
- 写出 rejection 时的 adjusted distribution

**4. INT4 Dequant 逻辑（10min）**
- 写出 unpack INT4 from INT32 的位操作
- 写出 dequant 公式: w_fp = (w_int - zero) * scale

**5. MoE All-to-All 通信（5min）**
- 画出 4 GPU、8 experts 的 dispatch + combine 流程
- 写出通信量公式

**6. Continuous Batching vs Static Batching（5min）**
- 画出两种方式的时间线对比
- 标注 GPU 利用率差异

### 自评

每项 0-2 分（0=不会, 1=部分正确, 2=完全正确）
总分 ≥ 8/12 通过

---

## 下午（2h）- 周检模拟面试

### 题库（随机抽 7 题，每题 5 分钟口答）

**Q1: FlashAttention 为什么比标准 attention 快？IO 复杂度差多少？**
- 标准: O(N^2) HBM IO（存储完整 attention matrix）
- FA: O(N^2 * d^2 / M)，当 M > d^2 时远小于 O(N^2)
- 本质：不存储 N×N matrix，用 online softmax 逐块计算

**Q2: vLLM 的 scheduler 怎么决定 preempt 哪个 request？**
- 默认 FIFO（最后加入的先被 preempt）
- 两种方式：swap（KV 换到 CPU）vs recompute（丢弃重算）
- 选择依据：有 CPU space → swap；否则 recompute

**Q3: Continuous batching 和 static batching 的区别？**
- Static: 等整个 batch 完成才处理下一个，利用率 30-50%
- Continuous: iteration-level scheduling，完成即替换，利用率 80-90%
- 吞吐提升 2-3x

**Q4: Decode attention 为什么是 memory-bound？怎么优化？**
- Q=[1,d], K=[seq,d] → GEMV，AI≈1
- 优化：batching（多 request 共享权重读取）、FlashDecoding（seq 维度并行）、KV FP8

**Q5: Speculative decoding 什么时候不值得用？**
- α < 0.5（频繁 reject）
- Draft model 太大（overhead 高）
- Batch size 大（decode 已接近 compute-bound）
- 短生成（overhead 占比大）

**Q6: GPTQ vs AWQ 核心区别？**
- GPTQ: reconstruction-based，逐列量化+误差补偿，慢但精度高
- AWQ: activation-aware scaling，保护重要权重，快且泛化好
- 选择：通用场景用 AWQ，固定 prompt 用 GPTQ

**Q7: FP8 为什么能 2x throughput？和 INT4 的区别？**
- FP8: Tensor Core 原生支持，直接做 8-bit 乘法 → 2x compute
- INT4: 存储 4-bit，但 dequant 到 fp16 再算 → 只省带宽不省计算
- FP8 适合 compute-bound（prefill），INT4 适合 memory-bound（decode）

**Q8: Chunked prefill 解决什么问题？代价？**
- 问题：长 prompt prefill 阻塞 decode → TTFT P99 飙升
- 解决：切成 512 token chunks，和 decode 交替
- 代价：prefill 总时间变长（kernel launch overhead）

**Q9: PagedAttention 的 block table 是什么？**
- 逻辑块 → 物理块的映射表（类似 OS 页表）
- 每个 sequence 有自己的 block table
- 好处：消除碎片，内存浪费 < 4%，支持 copy-on-write

**Q10: MoE 的 load balancing 为什么重要？怎么做？**
- 问题：如果所有 token 都去同一个 expert → 该 GPU 过载，其他空闲
- 解决：auxiliary loss 鼓励均匀分配 + capacity factor 限制每个 expert 的 token 数
- DeepSeek-V3 的方案：动态 top-k + 无 auxiliary loss（用 bias term 代替）

---

## 晚上（2h）- 阶段检 #1

### 模拟完整面试（100 分钟）

**环节 1: 项目深挖（20min）**

准备讲述你的 kernel lab 项目：
```
项目: LLM Kernel Optimization Lab
- 实现了 5 种 parallel reduction 优化（V1→V5，最终版用 warp shuffle）
- 实现了 FlashAttention Triton 版本，通过正确性测试
- 实现了 fused RMSNorm + residual kernel，加速 1.4x
- 所有 kernel 有 roofline 分析和 Nsight Compute profiling

量化成果:
- Reduction: V5 比 V1 快 Xx
- Fused RMSNorm: 比 unfused 快 1.4x，achieved BW = XX GB/s (XX% of peak)
- GEMM: Triton 版达到 cuBLAS 的 XX%
```

追问准备：
- 为什么选择 warp shuffle 而不是 shared memory？
- FlashAttention 的 rescaling 怎么保证数值正确？
- Fused kernel 的 occupancy 是多少？受什么限制？

**环节 2: 系统设计（25min）**

题目：设计一个支持 100 QPS 的 LLM serving 系统（7B 模型，平均输入 512 tokens，输出 256 tokens）

要点：
- 硬件选择：几张 A100？（计算 decode throughput）
- Batching 策略：continuous batching
- KV Cache 管理：PagedAttention
- 调度：prefill/decode 分离 or 混合
- 量化：FP8 or INT4？
- 监控：TTFT, TBT, throughput, GPU utilization

**环节 3: 八股/原理（15min，5 题）**

1. CUDA 线程层次：grid → block → warp → thread
2. Shared memory bank conflict 怎么避免？
3. ZeRO-3 的通信模式？
4. TP 中 AllReduce 插在哪里？
5. Online softmax 的数学推导？

**环节 4: 手撕 kernel（20min）**

题目：手写 fused softmax + scale kernel（Triton 或 CUDA 伪代码）

**环节 5: 算法题（20min）**

LRU Cache (146) 或 Top-K Frequent Elements (347)

### 自评打分

每环节 20 分，总分 100，及格 70。
第一次阶段检目标：≥ 60 分（允许不及格，用于发现薄弱点）。

---

## 本周总结

```markdown
## Week 2 总结

### 完成情况
- [ ] FlashAttention Triton 实现 + 正确性测试
- [ ] vLLM scheduler 源码阅读 + 流程图
- [ ] Continuous batching / chunked prefill 理解
- [ ] Speculative decoding 算法理解
- [ ] PagedAttention kernel 理解
- [ ] INT4 dequant kernel 实现
- [ ] MoE Expert Parallelism 理解

### 日检通过率: __/6

### 周检得分: __/21

### 阶段检 #1 得分: __/100
  - 项目深挖: __/20
  - 系统设计: __/20
  - 八股原理: __/20
  - 手撕 kernel: __/20
  - 算法题: __/20

### 薄弱点（阶段检暴露的问题）:
-

### 第 3 周重点调整:
-
```

---

## 第 3-8 周概要

后续每天的详细计划将在前两周完成后根据实际进度调整。大方向：

**Week 3**: SGLang 源码 + PD 分离 + Attention 变体 (GQA/MLA) + 项目启动
**Week 4**: Mini inference engine 或 开源 PR + 分布式实操
**Week 5**: 项目完善 + benchmark 图表 + 技术博客
**Week 6**: 系统设计专项 + 论文串讲练习
**Week 7**: Mock interview 密集训练 + 项目包装
**Week 8**: 简历定稿 + 投递 + 查漏补缺
