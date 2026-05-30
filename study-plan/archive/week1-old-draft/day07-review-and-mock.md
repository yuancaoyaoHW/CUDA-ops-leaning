# Day 7：周复习 + 周检模拟面试

## 学习目标
- 闭卷手写验证本周所有核心内容
- 模拟面试检验口头表达能力
- 预读下周材料

---

## 上午（2h）- 闭卷手写马拉松

### 规则
- 关掉所有资料、浏览器、笔记
- 打开空白编辑器
- 计时 45 分钟完成以下 7 项
- 完成后对照笔记自评，标记红色（不会）/黄色（模糊）/绿色（正确）

### 题目

**1. Warp Shuffle Reduction（5min）**
- 写出完整 CUDA kernel：输入 float* input, int n，输出 float* output（每个 block 一个结果）
- 包含 warp-level reduction + cross-warp reduction

**2. Online Softmax 数学推导（5min）**
- 写出一遍扫描的更新公式
- 解释为什么 rescaling 正确（一句话证明）

**3. Tiled GEMM 核心循环（10min）**
- 伪代码：shared memory 声明、load、syncthreads、compute
- 标注 block size 和 thread 分工

**4. Fused RMSNorm + Residual Triton Kernel（10min）**
- 完整 kernel：输入 X, Residual, Weight，输出 Out, NewResidual
- 包含 variance 计算和 rsqrt

**5. ZeRO-1/2/3 内存公式（5min）**
- 写出每种方案的 per-GPU 内存占用
- 写出通信量对比

**6. Tensor Parallelism 通信（5min）**
- 画出 MLP 的 Column Parallel + Row Parallel
- 标注 AllReduce 的位置和通信量

**7. Ring AllReduce 通信图（5min）**
- 画出 4 GPU 的 ReduceScatter 过程（3 步）
- 写出总通信量公式

### 自评标准

| 项目 | 绿色 | 黄色 | 红色 |
|------|------|------|------|
| 1 | 逻辑完全正确 | 有 1-2 个小错 | 写不出来 |
| 2 | 公式正确+证明 | 公式对但证明不清 | 公式错 |
| 3 | 结构完整 | 缺少 syncthreads 或 load 逻辑 | 写不出 |
| 4 | 可运行 | 有 bug 但思路对 | 写不出 |
| 5 | 公式全对 | 1-2 个记错 | 大部分不记得 |
| 6 | 图+公式正确 | 图对但公式不确定 | 画不出 |
| 7 | 步骤正确 | 方向对但细节错 | 不会 |

**红色项**：下周 Day 8 上午第一件事补回来。

---

## 下午（2h）- 周检模拟面试

### 操作方式

1. 从下面 10 题中随机选 7 题（可以用随机数生成器）
2. 每题口头回答 5 分钟（录音或对着镜子）
3. 自评 0-3 分
4. 总分 ≥ 15 通过

### 题库

**Q1: Bank conflict 是什么？在 GEMM 中怎么产生的？怎么避免？**

期望答案要点：
- 32 个 bank，4 bytes/bank
- 同一 warp 多个 thread 访问同一 bank 不同地址 → 串行化
- GEMM 中：Bs[kk][tx] 当多个 thread 的 tx 映射到同一 bank
- 避免：padding（+1）、swizzle、转置

**Q2: Softmax 是 compute-bound 还是 memory-bound？怎么优化？**

期望答案：
- Memory-bound，AI ≈ 1.25 FLOP/byte
- 单独优化意义有限（已接近 BW 峰值）
- 真正优化：fuse 到 attention（FlashAttention），避免写回 N×N attention matrix

**Q3: Kernel fusion 对什么类型的 kernel 有效？为什么？**

期望答案：
- 对 memory-bound kernel 有效
- 原因：减少 global memory round-trip，数据在 register/shared memory 中复用
- 对 compute-bound kernel 无效（瓶颈不在内存）
- 例子：RMSNorm + residual fuse 后减少 40% 内存访问

**Q4: ZeRO-3 vs DDP 通信量差多少？为什么值得？**

期望答案：
- ZeRO-3 通信量 ≈ 1.5x DDP（多了 forward 的 AllGather）
- 值得因为内存从 16Φ 降到 16Φ/N
- 例：7B 模型 8 GPU，DDP 需要 112GB/卡（放不下），ZeRO-3 只需 14GB/卡

**Q5: TP 的 Column/Row Parallel 分别在哪通信？通信量？**

期望答案：
- Column Parallel（切 A 列）：不需要通信（GeLU 是 element-wise）
- Row Parallel（切 B 行）：需要 AllReduce（部分和相加）
- 每层 2 次 AllReduce（attention + MLP）
- 通信量 = 2 * 2*(N-1)/N * batch * seq * hidden * sizeof

**Q6: 1F1B 比 GPipe 好在哪？Bubble ratio 一样吗？**

期望答案：
- Bubble ratio 相同：(PP-1) / (PP-1 + num_microbatches)
- 1F1B 优势：显存更低
  - GPipe：需要存所有 micro-batch 的 activation（num_microbatches 份）
  - 1F1B：同时只存 PP-1 份 activation
- 代价：实现更复杂

**Q7: 8xA100 80GB 部署 70B fp16 模型怎么选并行策略？**

期望答案：
- 70B fp16 = 140GB 参数，单卡放不下
- 方案 1：TP=8（8 卡 tensor parallel）→ 每卡 17.5GB 参数 + KV cache
- 方案 2：TP=4, PP=2 → 每卡 17.5GB，但 PP 有 bubble
- 推荐 TP=8：A100 有 NVLink 600GB/s，TP 通信开销可接受
- 如果要服务多个请求：TP=8 + 多副本（多 node）

**Q8: Roofline 的 ridge point 是什么？你的 4060 大约是多少？**

期望答案：
- Ridge point = Peak Compute / Peak Bandwidth
- AI < ridge point → memory-bound; AI > ridge point → compute-bound
- 4060 Laptop: FP16 peak ~176 TFLOPS, BW ~256 GB/s → ridge ≈ 687 FLOP/byte
- A100: FP16 peak ~312 TFLOPS, BW ~2 TB/s → ridge ≈ 156 FLOP/byte
- H100: FP16 peak ~990 TFLOPS, BW ~3.35 TB/s → ridge ≈ 295 FLOP/byte

**Q9: Double buffering 解决什么问题？画出时间线。**

期望答案：
- 解决：global memory load latency 暴露在关键路径上
- 无 DB：|Load|Compute|Load|Compute| → 串行
- 有 DB：Load 和 Compute overlap → 总时间 ≈ max(Load, Compute) * N
- 需要 2x shared memory
- CUDA 实现：cp.async + pipeline；Triton 实现：num_stages > 1

**Q10: 小 batch GEMM 为什么慢？和大 batch 有什么本质区别？**

期望答案：
- 大 batch GEMM (M large): AI = 2MNK / ((MK+KN+MN)*2) ≈ M（当 M≈N≈K）→ compute-bound
- 小 batch GEMM (M=1, GEMV): AI = 2NK / ((K+KN+N)*2) ≈ 1 → memory-bound
- 本质区别：大 batch 时权重矩阵被多个 query 复用（数据复用率 = M）
- 小 batch 时权重矩阵读一次只用一次 → bandwidth 瓶颈
- 这就是 decode 阶段 batching 重要的原因

---

## 晚上（1.5h）- 预读下周材料

### FlashAttention 论文预读

读 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" 的：
- Section 1: Introduction（为什么标准 attention 慢）
- Section 3.1: Algorithm（核心算法，重点看 Algorithm 1）
- Figure 1: IO 复杂度对比

### vLLM 论文预读

读 "Efficient Memory Management for Large Language Model Serving with PagedAttention" 的：
- Section 1: Introduction（KV cache 的内存浪费问题）
- Section 3: PagedAttention（block table 的设计）

### 记录问题

读完后写下 3-5 个不理解的问题，下周带着问题去实现和读源码。

---

## 本周总结模板

```markdown
## Week 1 总结

### 完成情况
- [ ] 5 个 reduction 版本 + benchmark
- [ ] Online softmax (Triton + CUDA)
- [ ] Tiled GEMM V1
- [ ] GEMM V2 (thread tiling + double buffering)
- [ ] Fused RMSNorm + Residual
- [ ] Triton GEMM
- [ ] Roofline 总结表
- [ ] LRU Cache

### 闭卷手写通过率
- Day 1: __/3
- Day 2: __/3
- Day 3: __/3
- Day 4: __/3
- Day 5: __/3
- Day 6: __/3

### 周检得分: __/21

### 红色项（需要补课）:
-

### 下周重点:
-
```
