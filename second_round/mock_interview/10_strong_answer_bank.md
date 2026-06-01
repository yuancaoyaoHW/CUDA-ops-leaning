# 强答案库 (Strong Answer Bank)

候选人背景：浙大硕士+西交本科，vLLM-Ascend PR#1032（EAGLE-3 proposer，吞吐+55%），RAG后端独立负责（RAGAS 90%）

---

## LLM Inference（25 条）

**Q1: PagedAttention 的核心思想是什么？解决了什么问题？**
**A:** PagedAttention 将 KV cache 按固定大小的 block（通常 16 tokens）管理，类似 OS 虚拟内存分页。解决了传统实现中 KV cache 预分配导致的 60-80% GPU 内存浪费问题。vLLM 论文显示，PagedAttention 将有效 batch size 提升 2-4x，吞吐提升 2-4x，内存浪费降至 <4%。工程上需要维护 block table 做 logical→physical 映射，copy-on-write 支持 beam search 共享 prefix blocks。

**Q2: Continuous batching 与 static batching 的区别和收益？**
**A:** Static batching 等所有请求完成才释放 slot，短请求被长请求阻塞，GPU utilization 仅 30-50%。Continuous batching（iteration-level scheduling）每个 decode step 可插入新请求、移除已完成请求。Orca 论文显示吞吐提升 36x（极端场景）。工程 tradeoff：调度开销增加约 0.1ms/iteration，需要 preemption 机制处理内存不足时的请求 swap/recompute。

**Q3: Speculative decoding 的原理和适用场景？**
**A:** 用小模型（draft model）自回归生成 K 个 token（通常 K=3-5），大模型一次 forward pass 并行验证。接受率 α 通常 70-85%，有效将 TPOT 降低为 1/(α*K) 倍。适用于 decode-bound 场景（batch size 小、模型大）。我在 vLLM-Ascend 实现的 EAGLE-3 proposer 将吞吐从 9.22 提升到 14.30 tok/s（+55%），TPOT 降低 39%，关键是 draft model 的 overhead 必须 <20% target model latency。

**Q4: KV cache 的内存占用如何计算？**
**A:** KV cache per token = 2 * num_layers * num_heads * head_dim * dtype_bytes。以 LLaMA-70B 为例：2 * 80 * 64 * 128 * 2(FP16) = 2.62MB/token。2048 token 序列需要 5.24GB。batch=32 时需要 167GB，超过单卡 80GB，必须用 TP 或 PagedAttention 减少浪费。这是 LLM serving 的核心内存瓶颈。

**Q5: Prefill 和 decode 阶段的计算特性有何不同？**
**A:** Prefill 是 compute-bound：处理整个 prompt 的矩阵乘（seq_len * hidden），GPU utilization 高达 70-90%，受 FLOPS 限制。Decode 是 memory-bound：每步只生成 1 token，GEMV 操作，arithmetic intensity <1，GPU utilization 仅 5-15%，受 HBM bandwidth 限制。这导致 disaggregated serving（prefill/decode 分离部署）成为优化方向，Splitwise 论文显示可降低 20-40% 成本。

**Q6: vLLM 的调度策略是怎样的？**
**A:** vLLM 使用 FCFS + priority-based scheduling。每个 iteration：1）先调度 running 队列中的 decode 请求；2）按 arrival time 从 waiting 队列取 prefill 请求填满剩余 GPU memory。当内存不足时触发 preemption：swap（KV cache 写到 CPU）或 recompute（丢弃 KV cache 后重算）。swap 适合长序列（避免重算开销），recompute 适合短序列（避免 PCIe 带宽瓶颈，PCIe 4.0 仅 32GB/s）。

**Q7: INT8/INT4 量化对推理性能的影响？**
**A:** W8A8 量化将模型体积减半，decode 阶段（memory-bound）吞吐近似翻倍，prefill 阶段收益较小（约 1.3-1.5x）。W4A16（GPTQ/AWQ）进一步压缩但需要 dequantize 开销。精度损失：W8A8 通常 <0.5% perplexity 退化，W4A16 约 1-2%。工程上需要 calibration dataset、per-channel vs per-tensor 选择、outlier channel 处理（SmoothQuant 将 activation outlier 迁移到 weight）。

**Q8: Tensor Parallelism 和 Pipeline Parallelism 的 tradeoff？**
**A:** TP 将每层的权重切分到多卡，每层需要 2 次 AllReduce（forward+backward 各一次），通信量 = 2 * hidden_size * dtype_bytes * (N-1)/N。适合同节点 NVLink（900GB/s）。PP 将不同层分配到不同卡，通信量仅为 activation（batch * seq * hidden），但引入 pipeline bubble（1/num_stages 效率损失）。实践中 TP=8（单节点）+ PP=N（跨节点）是标准配置。

**Q9: TTFT 和 TPOT 分别受什么因素影响？如何优化？**
**A:** TTFT（Time To First Token）= prefill latency，受 prompt 长度、compute capacity、排队时间影响。优化：prefix caching（命中率可达 60-80%）、chunked prefill、更多 prefill 实例。TPOT（Time Per Output Token）= decode latency/token，受 batch size、memory bandwidth、KV cache 大小影响。优化：speculative decoding（我的 EAGLE-3 实现降低 39%）、量化、更高 memory bandwidth 硬件。SLA 通常要求 TTFT<2s、TPOT<50ms。

**Q10: Prefix caching 的实现原理和收益？**
**A:** 对相同 prefix 的请求复用已计算的 KV cache。vLLM 用 hash(token_ids) 作为 block key，LRU 淘汰。收益取决于 prefix 命中率：system prompt 共享场景命中率 >90%，可将 TTFT 从秒级降到百毫秒级。工程挑战：hash 冲突处理、多租户隔离、cache 一致性（模型更新后需 invalidate）。RadixAttention（SGLang）用 radix tree 支持任意 prefix 匹配，比 vLLM 的 block-level 粒度更细。

**Q11: 如何设计 LLM serving 的 autoscaling 策略？**
**A:** 核心指标：queue depth（排队请求数）、GPU KV cache utilization、P99 TTFT。策略：当 queue_depth > threshold 或 KV cache usage > 85% 持续 30s 时 scale up，当 GPU utilization < 20% 持续 5min 时 scale down。冷启动问题：模型加载需要 30-120s（70B 模型从 S3 加载约 60s），需要 warm pool 预留 1-2 个 standby 实例。Scale down 需要 graceful drain：停止接收新请求，等待 running 请求完成。

**Q12: Chunked prefill 解决什么问题？**
**A:** 长 prompt 的 prefill 会独占 GPU 数百毫秒，阻塞 decode 请求导致 TPOT spike。Chunked prefill 将 prefill 拆成固定大小的 chunk（如 512 tokens），与 decode 请求交替执行。Sarathi-Serve 论文显示 P99 TPOT 降低 3-5x。Tradeoff：prefill 总时间略增（多次 kernel launch overhead），但 decode 请求的 tail latency 大幅改善。vLLM 0.4+ 已支持此特性。

**Q13: FlashAttention 的核心优化是什么？**
**A:** FlashAttention 通过 tiling 将 attention 计算分块到 SRAM（192KB on A100），避免将 O(N^2) 的 attention matrix 写回 HBM。IO 复杂度从 O(N^2) 降到 O(N^2*d/M)，其中 M 是 SRAM 大小。实测 A100 上 FlashAttention-2 达到 230 TFLOPS（理论峰值 312 TFLOPS 的 73%），比标准实现快 5-9x。关键技巧：online softmax（Milakov & Gimelshein 2018）避免两次 pass。

**Q14: 多租户 LLM serving 如何做资源隔离？**
**A:** 三层隔离：1）请求级：per-tenant rate limiting + priority queue，高优租户优先调度；2）内存级：per-tenant KV cache quota，防止单租户 OOM 影响全局；3）实例级：关键租户独占 GPU，普通租户共享池。Fairness 指标：per-tenant P99 latency 差异 <20%。实现上用 token bucket 限流 + weighted fair queuing 调度，类似网络 QoS。

**Q15: 如何评估 speculative decoding 的 draft model 选择？**
**A:** 关键指标：acceptance rate α 和 draft overhead ratio。α = 验证通过的 token 数 / draft 生成的 token 数，需要 >70% 才有收益。Draft overhead = draft_time / target_time，需要 <20%。选择策略：同架构小模型（如 7B draft for 70B target）、self-draft（Medusa 多头）、retrieval-based（EAGLE 系列用 feature 预测）。我实现的 EAGLE-3 在 Ascend 上 α≈80%，draft overhead≈15%。

**Q16: Disaggregated serving（prefill/decode 分离）的架构设计？**
**A:** Prefill 节点用高 FLOPS GPU（如 H100 SXM），decode 节点可用低成本 GPU（如 L4）或高 bandwidth 配置。KV cache 通过 RDMA/NVLink 从 prefill 传输到 decode 节点。Splitwise/DistServe 论文显示成本降低 20-40%。挑战：KV transfer latency（70B 模型 2048 token 的 KV≈5GB，RDMA 100Gbps 需要 400ms），需要 pipeline 化传输与 decode 重叠。

**Q17: 如何处理 long context（128K+ tokens）的推理？**
**A:** 挑战：KV cache 巨大（LLaMA-70B 128K token≈330GB）、attention 计算 O(N^2)。方案：1）Ring Attention 将 sequence 分布到多卡，通信与计算重叠；2）KV cache compression（H2O 保留 heavy hitter tokens，StreamingLLM 保留 sink tokens + recent window）；3）量化 KV cache（FP8 减半内存）。实践中 128K context 需要至少 4xH100（TP=4），TTFT 约 10-30s。

**Q18: vLLM 的 block manager 如何工作？**
**A:** Block manager 维护 free block pool + per-sequence block table。分配：请求到达时按需分配 block（lazy allocation），每个 block 存储 block_size 个 token 的 KV cache。释放：请求完成时归还所有 block。Fork：beam search 时 COW（copy-on-write），多个 beam 共享 prefix blocks，仅在写入时复制。Preemption：按 priority 选择 victim，swap 到 CPU 或标记为 recompute。

**Q19: 如何做 LLM 推理的 A/B testing？**
**A:** 流量分割：router 层按 user_id hash 分流到不同模型版本。指标：1）质量指标（human eval、LLM-as-judge）；2）性能指标（TTFT P50/P99、TPOT、throughput）；3）业务指标（completion rate、user satisfaction）。统计显著性需要足够样本量（通常 >1000 requests/variant）。Canary deployment：先 5% 流量验证性能无退化，再逐步放量。回滚条件：P99 latency 退化 >20% 或 error rate >0.1%。

**Q20: 如何优化 GPU utilization 在 LLM serving 中？**
**A:** 典型问题：decode 阶段 GPU compute utilization 仅 5-15%（memory-bound）。优化：1）增大 batch size（continuous batching 动态填充）；2）Speculative decoding 增加每步计算量；3）Chunked prefill 混合 prefill+decode 提高 arithmetic intensity；4）Multi-query/Grouped-query attention 减少 KV cache 内存占用，允许更大 batch。目标：GPU utilization >60%（prefill）、>30%（decode with large batch）。

**Q21: EAGLE-3 相比 EAGLE-1/2 的改进是什么？**
**A:** EAGLE-1 用单层 Transformer 做 feature-level draft，EAGLE-2 引入 dynamic draft tree 根据 confidence 调整树结构。EAGLE-3 进一步优化了 proposer 的 feature extraction，减少 draft overhead。我在 vLLM-Ascend 的实现中，EAGLE-3 proposer 在 Ascend 910B 上实现了 55% 吞吐提升（9.22→14.30 tok/s），关键优化包括：NPU 算子融合减少 kernel launch 开销、draft tree 的动态剪枝策略、KV cache 复用避免重复计算。

**Q22: 如何设计 model serving 的 health check 和 failover？**
**A:** Health check 三层：1）Liveness（进程存活，每 5s）；2）Readiness（模型加载完成，能处理请求）；3）Performance（P99 latency < SLA，每 30s）。Failover：检测到节点不健康后 10s 内将流量切走（DNS TTL 或 load balancer health check）。有状态问题：running 请求的 KV cache 丢失，需要 client retry + 重新 prefill。多副本部署确保单点故障不影响可用性，目标 99.9% availability。

**Q23: Guided decoding（structured output）如何实现？**
**A:** 在每步 decode 时用 FSM（finite state machine）或 grammar 约束 logits mask。JSON schema → regex → DFA，每步只允许合法 token。Outlines/LMFE 实现。性能影响：mask 计算 <1ms/step（预编译 DFA），几乎无 overhead。挑战：复杂 grammar 的 DFA 状态爆炸（需要 lazy evaluation）、与 speculative decoding 兼容（draft tokens 也需要满足约束）。

**Q24: 如何做 LLM serving 的 cost optimization？**
**A:** 成本构成：GPU 租赁（70-80%）、网络/存储（10-15%）、运维（5-10%）。优化：1）Spot instance 用于非 SLA 流量（节省 60-70%）；2）量化降低 GPU 需求（W8A8 减半 GPU 数）；3）Prefix caching 减少重复计算（节省 30-50% compute）；4）Right-sizing：根据 traffic pattern 选择 GPU 型号（decode-heavy 用高 bandwidth GPU，prefill-heavy 用高 FLOPS GPU）。目标：cost/1M tokens < $1（output token，70B 模型）。

**Q25: Request-level scheduling vs iteration-level scheduling 的区别？**
**A:** Request-level：整个请求绑定到一个 batch slot 直到完成，短请求等长请求（head-of-line blocking）。Iteration-level（continuous batching）：每个 decode iteration 独立调度，完成的请求立即释放 slot。性能差异：在输出长度方差大时（如 10-2000 tokens），iteration-level 吞吐提升 5-20x。实现复杂度：需要动态 memory 管理（PagedAttention）、per-iteration 调度决策、preemption 支持。

---

## CUDA/GPU（25 条）

**Q1: GPU memory hierarchy 各层的带宽和延迟？**
**A:** HBM（全局内存）：A100 2TB/s 带宽，400-600 cycle 延迟，80GB 容量。L2 cache：5TB/s，200 cycle，40MB。L1/Shared Memory（SMEM）：19TB/s，20-30 cycle，每 SM 192KB（A100 可配置 L1/SMEM 比例）。Register：无延迟，每 SM 256KB。优化核心：将热数据从 HBM 提升到 SMEM/Register，减少全局内存访问。Roofline 模型中，arithmetic intensity < 机器的 ops:byte 比时为 memory-bound。

**Q2: 什么是 memory coalescing？为什么重要？**
**A:** 同一 warp（32 threads）的内存访问如果落在连续的 128-byte 段内，硬件合并为一次 transaction。非 coalesced 访问会产生多次 transaction，浪费带宽。例如：stride-2 访问只利用 50% 带宽，stride-32 退化为 32 次独立 transaction。优化：确保 thread i 访问 array[base + i]（连续模式），对 AoS 数据考虑转为 SoA 布局。实测 coalesced vs non-coalesced 可差 10-20x 性能。

**Q3: Shared memory 的 bank conflict 是什么？如何避免？**
**A:** Shared memory 分为 32 个 bank（每 bank 4 bytes 宽），同一 warp 中多个 thread 访问同一 bank 的不同地址会串行化（N-way conflict = N 倍延迟）。避免方法：1）padding（如 `__shared__ float s[32][33]` 加一列避免列访问 conflict）；2）调整访问模式使 thread i 访问 bank i；3）broadcast（所有 thread 访问同一地址不冲突）。Nsight Compute 的 shared memory efficiency 指标可诊断。

**Q4: Warp divergence 的性能影响和处理方式？**
**A:** 同一 warp 内 thread 走不同分支时，硬件串行执行所有分支（inactive thread masked），执行时间 = 所有分支时间之和。影响：最坏情况 32 个分支 = 32x 性能退化。优化：1）将分支条件与 warp 对齐（如 `if(threadIdx.x / 32 < threshold)`）；2）数据预排序使同一 warp 走相同路径；3）用 predication 替代短分支。实际中 divergence 通常导致 10-30% 性能损失。

**Q5: Occupancy 是什么？高 occupancy 一定好吗？**
**A:** Occupancy = active warps / max warps per SM。受限于三个因素：registers per thread、shared memory per block、threads per block。高 occupancy 有助于隐藏 memory latency（更多 warp 可切换），但不一定最优：有时低 occupancy + 更多 register（减少 spill to local memory）性能更好。经验：memory-bound kernel 追求高 occupancy（>50%），compute-bound kernel 可接受较低 occupancy（25-50%）换取更多 register。

**Q6: Roofline model 如何指导 kernel 优化？**
**A:** Roofline 将 kernel 性能上界建模为 min(peak_FLOPS, peak_bandwidth * arithmetic_intensity)。AI = FLOPs / Bytes_accessed。A100：312 TFLOPS FP16，2TB/s HBM → ridge point = 156 FLOPs/byte。AI < 156 为 memory-bound（优化数据复用、减少访存），AI > 156 为 compute-bound（优化指令吞吐、用 Tensor Core）。GEMM 的 AI = O(N)（随矩阵增大），element-wise 的 AI = O(1)（永远 memory-bound）。

**Q7: GEMM tiling 优化的核心思想？**
**A:** 将大矩阵乘分块：每个 thread block 负责输出矩阵的一个 tile（如 128x128），从 HBM 加载 A 和 B 的子块到 shared memory，在 SMEM 中做小矩阵乘。数据复用率 = tile_size / block_size。双缓冲（double buffering）：一组 SMEM buffer 计算时，另一组异步加载下一块，隐藏 memory latency。CUTLASS 实现的分层 tiling：thread block tile → warp tile → thread tile，每层最大化对应层级的数据复用。

**Q8: Parallel reduction 的优化技巧？**
**A:** 基础版：每步一半 thread 做加法，log2(N) 步完成。优化：1）避免 warp divergence（用 stride 从大到小而非从小到大）；2）第一步从 global memory 加载时就做部分 reduce（减少 kernel launch）；3）最后 32 个元素用 warp shuffle（`__shfl_down_sync`）避免 shared memory 开销；4）grid-stride loop 处理大数组。单 block 1024 threads 可 reduce 2048 元素，多 block 需要二次 reduce 或 atomic。

**Q9: 如何实现高效的 softmax kernel？**
**A:** 三步：1）求 max（数值稳定）；2）求 exp(x-max) 的 sum；3）归一化。朴素实现需要 3 次遍历数据（3x HBM 读取）。优化：online softmax 一次遍历同时维护 running max 和 running sum（FlashAttention 的核心）。对于 row-wise softmax（attention），每行分配一个 warp，用 warp shuffle 做 reduce。大行（>1024）需要多 warp 协作 + shared memory。目标：达到 HBM bandwidth 上限（memory-bound kernel）。

**Q10: Tensor Core 的使用条件和性能收益？**
**A:** A100 Tensor Core：每 cycle 执行 256 FP16 FMA（16x16x16 矩阵乘），峰值 312 TFLOPS vs CUDA Core 19.5 TFLOPS（16x 差距）。使用条件：矩阵维度必须是 8/16 的倍数，数据需要特定 layout（row-major A, col-major B）。通过 WMMA API 或 MMA PTX 指令调用。实际利用率取决于数据搬运效率：如果 SMEM→Register 带宽不足，Tensor Core 会 stall。CUTLASS 的 warp-level MMA 抽象简化了使用。

**Q11: Nsight Compute 中最重要的性能指标有哪些？**
**A:** 1）SM throughput（计算利用率）；2）Memory throughput（HBM 带宽利用率）；3）Achieved occupancy（实际 vs 理论）；4）Warp stall reasons（memory dependency、execution dependency、barrier）；5）L1/L2 hit rate；6）Shared memory efficiency（bank conflict 程度）。诊断流程：先看 roofline 判断 bound 类型，再看 stall reason 定位瓶颈。memory-bound kernel 关注 coalescing 和 cache hit，compute-bound 关注 instruction mix 和 Tensor Core 利用。

**Q12: CUDA stream 和 event 的作用？**
**A:** Stream 是 GPU 上的有序命令队列，不同 stream 间可并行执行。用途：1）overlap compute 和 memory transfer（H2D/D2H 与 kernel 并行）；2）多 kernel 并行（小 kernel 无法占满 GPU）。Event 用于 stream 间同步和精确计时。实践：推理框架用多 stream pipeline 化 prefill 和 decode；训练用 stream 重叠 gradient allreduce 和 backward compute。注意：同一 stream 内严格有序，跨 stream 需要显式同步。

**Q13: GPU 的 warp scheduler 如何工作？**
**A:** 每个 SM 有 4 个 warp scheduler（A100），每 cycle 每个 scheduler 选择一个 eligible warp 发射指令。Eligible = 操作数就绪 + 无 structural hazard。Latency hiding：当一个 warp 等待内存返回（400+ cycles）时，scheduler 切换到其他 ready warp（零开销切换，因为每个 warp 有独立 register file）。这就是为什么高 occupancy 有助于隐藏 latency：更多 warp = 更多切换选择。

**Q14: 如何优化 GPU kernel 的 register 使用？**
**A:** Register 过多 → occupancy 下降（每 SM 65536 registers，1024 threads 时每 thread 最多 64）。Register spill → local memory（实际在 HBM，极慢）。优化：1）`__launch_bounds__(maxThreads, minBlocks)` 提示编译器；2）减少 live variables（重新计算 vs 存储）；3）用 shared memory 替代部分 register 需求；4）`-maxrregcount` 编译选项强制限制。Nsight 中 local memory traffic > 0 说明有 spill。

**Q15: Atomic 操作的性能特点和替代方案？**
**A:** Atomic 在 global memory 上序列化同一地址的访问，高竞争时性能极差（thousands of cycles）。L2 cache 中的 atomic 较快（A100 支持）。替代方案：1）Hierarchical reduction：先 warp-level shuffle reduce，再 block-level shared memory reduce，最后一个 atomic 写全局；2）Privatization：每个 block 维护 local copy，最后合并；3）Segmented scan。Histogram 场景：privatization + shared memory histogram 可比 naive atomic 快 10-50x。

**Q16: cudaMalloc vs cudaMallocAsync 的区别？**
**A:** cudaMalloc 同步分配，每次调用 overhead 约 100-500μs（涉及 driver 调用）。cudaMallocAsync（CUDA 11.2+）使用 memory pool，异步分配/释放，overhead <1μs。Stream-ordered memory allocator 自动复用 freed memory，减少碎片。推理框架中频繁分配/释放 workspace 时，pool allocator 可将 allocation overhead 从 ms 级降到 μs 级。PyTorch 的 caching allocator 类似思想。

**Q17: 如何实现高效的 transpose kernel？**
**A:** Naive transpose：thread(i,j) 读 input[i][j] 写 output[j][i]，读 coalesced 但写 non-coalesced（stride = N）。优化：用 shared memory 做中转——每个 block 读一个 tile 到 SMEM（coalesced read），然后从 SMEM 转置后写回（coalesced write）。关键：SMEM 声明为 `[TILE][TILE+1]`（+1 padding 避免 bank conflict）。性能可达 HBM 带宽的 90%+，接近 memory copy 速度。

**Q18: CUDA kernel launch 的 overhead 是多少？如何减少？**
**A:** Kernel launch overhead 约 3-10μs（CPU 端提交到 GPU 队列）。对于执行时间 <10μs 的小 kernel，launch overhead 占比显著。减少方法：1）Kernel fusion（合并多个小 kernel）；2）CUDA Graphs（预录制 kernel 序列，一次 launch 执行整个 graph，overhead 降到 <1μs）；3）Persistent kernel（kernel 内循环处理多批数据）。推理框架中 CUDA Graphs 可将 decode 阶段 overhead 从 ms 级降到 μs 级。

**Q19: 如何分析 kernel 是 compute-bound 还是 memory-bound？**
**A:** 方法 1：计算 arithmetic intensity（FLOPs / bytes），与硬件 ridge point 比较。方法 2：Nsight Compute 的 Speed of Light 面板，看 SM% vs Memory%——哪个接近 100% 就是瓶颈。方法 3：实验法——增加计算量（如循环展开）看是否变慢，或减少数据量看是否变快。大多数推理 kernel（layernorm、softmax、activation）是 memory-bound，GEMM 是 compute-bound。

**Q20: Grid-stride loop 模式的优势？**
**A:** 模式：`for(int i = blockIdx.x*blockDim.x+threadIdx.x; i < N; i += gridDim.x*blockDim.x)`。优势：1）一个 kernel 处理任意大小数据（不需要 N/blockDim 个 block）；2）grid 大小可固定为 SM 数量 * occupancy，最大化 GPU 利用率；3）减少 tail effect（最后几个 block 利用率低）；4）便于 persistent kernel 设计。适用于 element-wise 操作和 reduction 的第一阶段。

**Q21: FP16 vs BF16 在 GPU 计算中的区别？**
**A:** FP16：5 bit exponent + 10 bit mantissa，范围 ±65504，精度高但容易 overflow。BF16：8 bit exponent + 7 bit mantissa，范围同 FP32（±3.4e38），精度低但不易 overflow。训练中 BF16 更稳定（不需要 loss scaling），推理中 FP16 精度更好。A100 两者 Tensor Core 吞吐相同（312 TFLOPS）。H100 新增 FP8（E4M3/E5M2），吞吐翻倍到 989 TFLOPS。

**Q22: 如何实现 warp-level primitives 优化？**
**A:** Warp shuffle（`__shfl_down_sync`, `__shfl_xor_sync`）允许 warp 内 thread 直接交换 register 数据，无需 shared memory。用途：1）Warp reduce（5 步 shuffle 完成 32 元素 reduce）；2）Broadcast（一个 thread 的值广播到全 warp）；3）Butterfly pattern（FFT-like 通信）。性能：1 cycle latency vs shared memory 的 20+ cycles。限制：只能在同一 warp 内通信，跨 warp 仍需 shared memory。

**Q23: CUDA 中如何处理 branch predication vs actual branching？**
**A:** 短分支（<7 条指令）编译器自动用 predication：两个分支都执行，用 predicate register 选择结果写入。无 divergence 开销但浪费计算。长分支用实际 branch 指令：divergent warp 串行执行各分支。优化建议：保持分支体短小让编译器 predicate；对长分支重组数据使 warp 内 uniform；用 `__ballot_sync` 检测 divergence 程度做动态决策。

**Q24: Multi-GPU 通信中 NVLink vs PCIe 的性能差异？**
**A:** NVLink（A100）：600GB/s 双向，延迟 ~1μs。PCIe 4.0：64GB/s 双向，延迟 ~5μs。NVLink 带宽是 PCIe 的 9.4x。影响：Tensor Parallelism 的 AllReduce 通信量 = 2 * hidden_size * batch * seq * dtype_bytes，70B 模型 TP=8 每步约 100MB，NVLink 下 <0.2ms，PCIe 下 ~1.6ms。因此 TP 必须在 NVLink 连接的 GPU 间，跨节点只能用 PP 或 DP。

**Q25: 如何用 Nsight Systems 分析 end-to-end 推理性能？**
**A:** Nsight Systems 提供 timeline 视图：CPU 调用、CUDA API、kernel 执行、memory transfer 全部可视化。分析步骤：1）看 GPU idle gap（CPU overhead 或同步等待）；2）看 kernel 重叠度（多 stream 是否有效并行）；3）识别 long-tail kernel（优化热点）；4）检查 HtoD/DtoH transfer 是否与 compute 重叠。推理场景关注：prefill 的 GEMM kernel 占比、decode 的 kernel launch overhead、attention kernel 的执行时间。

---

## RAG Infrastructure（20 条）

**Q1: Chunking 策略如何选择？对检索质量的影响？**
**A:** 常见策略：fixed-size（512 tokens，overlap 50-100）、semantic（按段落/标题分割）、recursive（先大块再细分）。Chunk 太大→检索精度下降（噪声多），太小→上下文不完整。实测：512 tokens + 50 overlap 在通用 QA 上 recall@10 最优。我的项目中用 recursive splitter + metadata 保留父子关系，RAGAS faithfulness 从 0.78 提升到 0.90。关键：chunk 边界不能切断关键信息，需要 sentence-aware splitting。

**Q2: Embedding 模型选择的考量因素？**
**A:** 维度 vs 性能 tradeoff：768d（BERT-base）检索速度快但精度一般，1024d（BGE-large）精度高但存储和计算成本增加。多语言需求选 multilingual-e5。评估指标：MTEB benchmark 的 retrieval subset。延迟要求：batch encoding 1000 chunks，768d 约 2s（A10G），1536d（OpenAI ada-002）需要 API 调用约 5s。我的项目用 BGE-large-zh，recall@10 达 92%，encoding throughput 500 chunks/s。

**Q3: Vector search 的 ANN 算法选择？**
**A:** HNSW：recall@10 >95%（ef_search=128），延迟 <5ms（1M vectors），内存占用高（每 vector 额外 ~1KB 图结构）。IVF-PQ：内存低（PQ 压缩 32x），但 recall 降到 85-90%，适合 >10M vectors。Flat（暴力搜索）：100% recall 但 >100ms（1M vectors）。选择依据：<1M vectors 用 HNSW，>10M 用 IVF-HNSW 或 ScaNN。我的项目 50K documents 用 HNSW（Milvus），P99 检索延迟 <10ms。

**Q4: Hybrid search（向量+关键词）如何实现？**
**A:** 架构：BM25（keyword）+ dense retrieval（embedding）并行检索，用 RRF（Reciprocal Rank Fusion）或 linear combination 合并。RRF 公式：score = Σ 1/(k+rank_i)，k=60 是常用值。收益：hybrid 比纯 dense 在 entity-heavy query 上 recall 提升 10-15%。实现：Elasticsearch（BM25）+ Milvus（dense），或用 Weaviate/Qdrant 内置 hybrid。我的项目 hybrid search 将 RAGAS context_recall 从 0.85 提升到 0.92。

**Q5: Reranker 的作用和性能影响？**
**A:** 两阶段检索：第一阶段 bi-encoder 快速召回 top-50（<10ms），第二阶段 cross-encoder rerank 到 top-5（精排）。Cross-encoder 精度高但慢（每对 query-doc 需要一次 forward pass）。延迟：rerank 50 docs 约 100-200ms（BGE-reranker-large on GPU）。收益：rerank 后 NDCG@5 提升 15-25%。优化：用 FP16 量化 reranker、batch inference、设置 top-k 上限。我的项目 rerank top-20→top-5，answer relevancy 提升 12%。

**Q6: Metadata filtering 如何提升检索精度？**
**A:** 在向量搜索前/后加 metadata 过滤（时间、来源、类别、权限）。Pre-filter：先过滤再搜索，减少搜索空间但可能 recall 不足。Post-filter：先搜索再过滤，recall 高但浪费计算。Milvus/Qdrant 支持 hybrid（filter 集成到 ANN 搜索中）。实测：加 time_range filter 后，时效性问题的 answer correctness 提升 20%。关键：metadata schema 设计要覆盖常见 query pattern。

**Q7: RAGAS 评估框架的核心指标？**
**A:** 四个核心指标：1）Faithfulness（答案是否基于 context，防幻觉）；2）Answer Relevancy（答案是否回答了问题）；3）Context Recall（检索是否覆盖了 ground truth）；4）Context Precision（检索结果中相关文档的排序）。我的项目最终指标：faithfulness 0.90、answer_relevancy 0.88、context_recall 0.92、context_precision 0.85。评估需要 ground truth dataset（至少 100+ QA pairs），用 LLM-as-judge 自动评分。

**Q8: 如何处理 RAG 中的数据新鲜度（freshness）问题？**
**A:** 增量更新 pipeline：1）Change Data Capture（CDC）监听数据源变更；2）增量 re-embedding + upsert 到向量库；3）过期文档标记 TTL 或 soft delete。延迟要求：实时场景 <5min（streaming pipeline），非实时 <1h（batch）。我的项目用 scheduled job 每 30min 增量同步，文档带 updated_at timestamp，检索时 boost 新文档权重。挑战：embedding 模型更新时需要全量 re-index。

**Q9: RAG 系统如何 scale 到百万级文档？**
**A:** 分层架构：1）存储层：分布式向量库（Milvus cluster，sharding by collection）；2）计算层：embedding service 水平扩展（GPU pod autoscaling）；3）缓存层：热门 query 结果缓存（Redis，TTL 5min）。性能目标：1M docs 检索 P99 <50ms，embedding throughput >1000 docs/s。Milvus 分片策略：按 tenant 或 time range 分 collection，避免单 collection 过大（>10M vectors 性能下降）。

**Q10: Query transformation 技术有哪些？**
**A:** 1）Query rewriting：LLM 改写用户 query 为更适合检索的形式；2）HyDE（Hypothetical Document Embedding）：LLM 生成假设答案，用其 embedding 检索；3）Multi-query：生成多个 query 变体，合并检索结果；4）Step-back prompting：生成更抽象的 query 获取背景知识。实测 HyDE 在 ambiguous query 上 recall 提升 15-20%，但增加一次 LLM 调用延迟（200-500ms）。我的项目用 query rewriting，recall 提升 8%。

**Q11: RAG vs Fine-tuning 的选择标准？**
**A:** RAG 适合：知识频繁更新、需要引用来源、数据量大（>10K docs）、多领域。Fine-tuning 适合：固定知识、需要特定风格/格式、低延迟要求（省去检索步骤）。混合方案：fine-tune 基础能力 + RAG 补充实时知识。成本对比：RAG 运行时成本高（检索+长 context），fine-tuning 一次性训练成本高但推理便宜。我的项目选 RAG 因为文档每周更新，fine-tuning 无法跟上。

**Q12: 如何处理 RAG 中的多跳推理（multi-hop）？**
**A:** 单次检索无法回答需要多步推理的问题。方案：1）Iterative retrieval：LLM 生成中间 query，多轮检索；2）Graph RAG：构建知识图谱，沿关系路径检索；3）Chain-of-thought + retrieval：每步推理后检索验证。延迟 tradeoff：多跳增加 2-5x 延迟。实测：2-hop 问题上 iterative retrieval 比 single-shot recall 提升 30%。限制：超过 3 hop 准确率急剧下降，需要 fallback 到 human-in-the-loop。

**Q13: 向量数据库的一致性和可用性如何保证？**
**A:** CAP tradeoff：大多数向量库选择 AP（最终一致性）。Milvus：segment sealed 后不可变（类 LSM-tree），写入先到 growing segment，定期 seal + build index。副本机制：每个 segment 2-3 副本分布在不同 query node。故障恢复：segment 数据持久化到 S3/MinIO，node 故障后从对象存储恢复。写入延迟：flush interval 决定可见性延迟（默认 1s）。我的项目配置 2 副本 + 1s flush，可用性 99.9%。

**Q14: 如何优化 RAG 的端到端延迟？**
**A:** 延迟分解：embedding query（20-50ms）+ vector search（5-20ms）+ rerank（100-200ms）+ LLM generation（500-2000ms）。优化：1）embedding 用 ONNX Runtime 加速（-40% latency）；2）减少 rerank candidates（50→20）；3）streaming LLM output（感知延迟降低）；4）并行化 retrieval 和 rerank。我的项目端到端 P50=800ms，P99=1.5s（含 LLM generation）。瓶颈在 LLM generation，检索链路优化空间有限。

**Q15: RAG 中如何处理表格和结构化数据？**
**A:** 挑战：表格 embedding 效果差（行列关系丢失）。方案：1）Table-to-text：LLM 将表格转为自然语言描述后 embedding；2）结构化查询：检测到表格相关 query 时转为 SQL/Pandas 查询；3）多模态 embedding：用 table-aware encoder。实测 table-to-text 在表格 QA 上 accuracy 提升 25% vs 直接 embedding。我的项目对 CSV/Excel 用 schema description + row sampling 生成 text chunk。

**Q16: 如何做 RAG 系统的 A/B testing？**
**A:** 分层实验：1）检索层（不同 chunking/embedding/rerank 策略）；2）生成层（不同 prompt/model）。指标：online（user satisfaction、click-through）+ offline（RAGAS metrics on golden set）。流量分割：按 session_id hash 保证同一用户体验一致。统计显著性：至少 500 queries/variant，用 bootstrap confidence interval。我的项目每次改动先跑 offline eval（200 QA pairs），RAGAS 提升 >2% 才上线 A/B test。

**Q17: Parent-child chunking 策略的实现？**
**A:** 思路：检索用小 chunk（精确匹配），返回给 LLM 用大 chunk（完整上下文）。实现：文档先切大块（parent，2000 tokens），再切小块（child，200 tokens），child 保留 parent_id。检索时匹配 child，返回对应 parent 给 LLM。收益：兼顾检索精度和上下文完整性，answer faithfulness 提升 10-15%。我的项目用 3 层（document→section→paragraph），检索 paragraph 返回 section。

**Q18: 如何监控 RAG 系统的线上质量？**
**A:** 指标体系：1）检索质量：empty result rate、avg retrieval score、diversity；2）生成质量：hallucination rate（LLM-as-judge 采样检测）、user feedback（thumbs up/down）；3）系统指标：latency P50/P99、error rate、throughput。告警：hallucination rate >5% 或 empty result >10% 触发 PagerDuty。我的项目每天采样 50 queries 做自动 RAGAS 评估，周报 review 质量趋势。

**Q19: Embedding 模型的 fine-tuning 何时必要？**
**A:** 必要场景：领域术语多（医疗、法律）、通用模型 recall 不足（<80%）、语言特殊（小语种）。方法：contrastive learning（positive pairs from click log or LLM-generated），训练数据 >10K pairs。收益：domain-specific fine-tuning 通常提升 recall 5-15%。成本：需要 GPU 训练（A10G 几小时）+ 全量 re-index。我的项目通用模型 recall 已达 92%，未做 fine-tuning，但准备了 hard negative mining pipeline 备用。

**Q20: RAG 系统的安全性考虑？**
**A:** 威胁：1）Prompt injection via retrieved content（恶意文档注入指令）；2）数据泄露（跨租户检索到其他用户文档）；3）PII 暴露（检索结果含敏感信息）。防护：1）retrieved content 放在 system prompt 的 data section，与 instruction 隔离；2）metadata filter 强制 tenant_id 过滤；3）PII detection + masking pipeline。我的项目实现了 tenant-level ACL + PII redaction，安全审计通过。

---

## System Design（20 条）

**Q1: 设计一个支持 1000 QPS 的 LLM serving 系统？**
**A:** 架构：Load Balancer → Router（prefix-aware routing）→ GPU Worker Pool（prefill/decode 可分离）。容量估算：假设平均 TPOT=30ms，单卡 decode throughput≈200 tok/s，平均输出 200 tokens，单卡≈1 req/s。1000 QPS 需要约 1000 张 GPU（70B 模型 TP=8 则 125 个 8-GPU 节点）。优化：prefix caching 减少 30% prefill 计算、continuous batching 提升 3-5x 吞吐、W8A8 量化减半 GPU 需求。成本约 $500K/月（H100 on-demand）。

**Q2: 多租户 LLM 平台的隔离设计？**
**A:** 三层隔离：1）逻辑隔离：per-tenant API key + rate limit（token bucket，burst=2x quota）；2）资源隔离：priority queue（P0 租户独占 GPU pool，P1/P2 共享 pool with weighted fair scheduling）；3）数据隔离：KV cache 按 tenant 标记，prefix cache 不跨 tenant 共享（防信息泄露）。SLA 分级：P0 保证 P99 TTFT<1s，P1 保证 P99<3s，P2 best-effort。计费：per-token（input/output 分别计价）。

**Q3: LLM serving 的 autoscaling 策略设计？**
**A:** 指标选择：不用 CPU/GPU utilization（LLM 场景不准确），用 queue_depth 和 KV_cache_usage。Scale-up 条件：queue_depth > 10 持续 30s 或 KV_cache_usage > 85%。Scale-down：queue_depth=0 且 GPU_util < 10% 持续 5min。冷启动优化：warm pool 预留 2 个 standby 节点（模型已加载），scale-up 延迟从 120s 降到 10s。Predictive scaling：基于历史流量模式（如工作日 9am spike）提前 scale。

**Q4: Disaggregated prefill/decode 架构的设计？**
**A:** 动机：prefill 是 compute-bound（需要高 FLOPS），decode 是 memory-bound（需要高 bandwidth）。架构：Prefill cluster（H100 SXM，高 FLOPS）+ Decode cluster（可用 L4 或 A10G，高性价比）。KV transfer：prefill 完成后通过 RDMA 将 KV cache 传输到 decode 节点。挑战：transfer latency（70B 2K tokens KV≈5GB，100Gbps RDMA 需 400ms）。优化：pipeline transfer 与 decode 重叠、KV compression（FP8）。收益：成本降低 20-40%（DistServe 论文）。

**Q5: 如何设计 model serving 的灰度发布系统？**
**A:** 流程：Shadow（镜像流量不返回用户）→ Canary（5% 真实流量）→ Gradual rollout（25%→50%→100%）。每阶段检查：latency P99 退化 <10%、error rate <0.1%、quality metrics（LLM-as-judge 采样 100 requests）无退化。回滚条件：任一指标超阈值自动回滚（<30s 切换流量）。实现：Istio/Envoy 做流量分割，Prometheus + Grafana 监控，PagerDuty 告警。模型版本管理：每个版本独立 deployment，共享 GPU pool。

**Q6: GPU cluster 的故障处理设计？**
**A:** 故障类型：1）单卡 ECC error（每周 1-2 次/千卡）→ 自动 drain + 替换；2）节点宕机 → running 请求丢失，client retry + 重新 prefill；3）NVLink 故障 → TP group 整组不可用，需要 failover 到备用节点组；4）网络分区 → 检测后隔离故障域。设计：每个 TP group 有 standby 替补（模型已加载），故障切换 <30s。Checkpoint：长请求定期 checkpoint KV cache 到 CPU memory，故障后恢复而非重算。

**Q7: 如何设计 LLM 推理的 cost optimization 系统？**
**A:** 多层优化：1）硬件层：spot instance 用于非 SLA 流量（节省 60-70%），reserved instance 用于 baseline；2）模型层：小模型处理简单请求（router 按 query complexity 分流），大模型处理复杂请求；3）缓存层：semantic cache（相似 query 复用答案，命中率 10-30%）；4）调度层：off-peak 时段 batch 处理非实时请求。目标：cost/1M output tokens 从 $15（naive）降到 $3-5。

**Q8: 设计一个支持 streaming 输出的 LLM API？**
**A:** 协议：SSE（Server-Sent Events）或 WebSocket。架构：Client → API Gateway（认证+限流）→ Router → Worker。Worker 每生成一个 token 通过 SSE 推送。挑战：1）长连接管理（timeout 设置 5min，heartbeat 每 30s）；2）断线重连（client 带 last_token_id，server 从 KV cache 继续）；3）背压（client 消费慢时 buffer 上限 1000 tokens，超过断开）。监控：per-stream latency、active connections、drop rate。

**Q9: 如何设计 LLM serving 的 observability 系统？**
**A:** 三支柱：1）Metrics：TTFT/TPOT P50/P99、throughput（tokens/s）、queue depth、GPU util、KV cache usage、error rate（Prometheus + Grafana）；2）Traces：per-request trace（prefill time、decode time、scheduling wait、每层 kernel time）用 OpenTelemetry；3）Logs：structured log（request_id、model、input/output tokens、latency breakdown）。告警：TTFT P99 > 2x baseline 或 error rate > 0.5% 触发 PagerDuty。Dashboard 按 tenant/model/GPU 分维度。

**Q10: 如何设计支持多模型的 serving 平台？**
**A:** 架构：Model Registry（版本管理+metadata）→ Scheduler（根据模型大小分配 GPU）→ Worker Pool（异构 GPU）。挑战：1）GPU 碎片化（大模型需要连续 8 卡，小模型 1 卡）→ bin-packing 调度；2）模型切换开销（加载 70B 需要 60s）→ 热门模型常驻，冷门模型按需加载；3）资源竞争 → per-model quota + priority。实现：KServe/Triton + 自定义 scheduler，模型存储在 S3，加载用 tensorstore 并行读取。

**Q11: 如何设计 LLM 推理的 request routing？**
**A:** 路由策略：1）Prefix-aware routing：相同 system prompt 的请求路由到同一 worker（最大化 prefix cache 命中）；2）Load-aware routing：选择 queue depth 最小的 worker；3）Locality-aware routing：同一 session 的请求路由到同一 worker（复用 KV cache）。实现：consistent hashing（prefix hash → worker），fallback 到 least-loaded。收益：prefix-aware routing 将 cache 命中率从 20% 提升到 70%，TTFT 降低 50%。

**Q12: 设计一个 LLM 推理结果的缓存系统？**
**A:** 两层缓存：1）Exact cache：hash(model + prompt + params) → response，命中率低（5-10%）但零延迟；2）Semantic cache：embedding similarity > 0.95 时复用（命中率 15-30%，需要 embedding 计算开销 20ms）。失效策略：TTL（1h for factual，5min for real-time）+ 模型版本变更时全量失效。存储：Redis cluster（exact）+ 向量库（semantic）。注意：non-deterministic 请求（temperature>0）不缓存。

**Q13: 如何设计跨区域的 LLM serving？**
**A:** 架构：Global Load Balancer（GeoDNS）→ Regional clusters（每个 region 独立 GPU pool）。数据一致性：模型权重从 central S3 同步到各 region（eventual consistency，新版本部署延迟 <10min）。Failover：region 故障时 DNS 切换到最近 region（TTL=30s）。挑战：1）GPU 供给不均（某些 region 无 H100）→ 跨 region 路由 long-context 请求；2）合规（数据不出境）→ per-region 数据隔离。延迟：同 region <50ms network，跨 region 100-200ms。

**Q14: 如何设计 LLM serving 的容量规划？**
**A:** 输入：1）QPS 预测（历史趋势 + 业务增长）；2）请求特征（avg input/output tokens）；3）SLA 要求（TTFT P99 < 2s）。计算：单卡 throughput = f(model_size, batch_size, quantization)，benchmark 得到。所需 GPU 数 = peak_QPS * avg_output_tokens / per_GPU_throughput * safety_margin(1.3x)。Buffer：预留 30% 应对 burst + 故障。采购周期：GPU 交付 3-6 个月，需要提前规划。Spot 补充 peak 需求。

**Q15: 如何设计 prompt 管理和版本控制系统？**
**A:** 架构：Prompt Registry（Git-backed，版本化）→ Prompt Compiler（变量替换+模板渲染）→ Serving Layer（runtime 获取最新 prompt）。功能：1）版本管理（每次修改生成新版本，可回滚）；2）A/B testing（不同 prompt 版本分流）；3）监控（per-prompt-version 的质量指标）。实现：prompt 存储在 Git repo，CI/CD 部署到 config service，serving 层 poll 或 webhook 更新。缓存：prompt 编译结果缓存，避免每次请求重新渲染。

**Q16: 设计一个 LLM 推理的 batch processing 系统？**
**A:** 场景：非实时任务（文档摘要、数据标注）。架构：Job Queue（SQS/Kafka）→ Batch Scheduler → GPU Worker Pool。优化：1）按 input length 排序分 batch（减少 padding 浪费）；2）用 spot instance（可中断，checkpoint 进度）；3）大 batch size 最大化 throughput（不受 latency SLA 约束）。吞吐目标：70B 模型 W8A8，batch=256，throughput≈5000 tok/s/GPU。成本：spot H100 $2/h，处理 1M tokens 约 $0.4。

**Q17: 如何设计 LLM serving 的安全架构？**
**A:** 层次：1）网络层：VPC 隔离 + mTLS（service mesh）；2）认证层：API key + OAuth2（per-tenant）；3）输入安全：prompt injection detection（classifier，准确率 >95%）+ content filter（toxicity/PII）；4）输出安全：output filter（hallucination detection、PII masking）；5）审计：所有请求 log 保留 90 天（加密存储）。合规：SOC2 要求加密 at-rest + in-transit，GDPR 要求数据删除能力。

**Q18: 如何设计支持 function calling 的 LLM serving？**
**A:** 架构：LLM Worker 生成 function call → Orchestrator 执行 tool → 结果注入 context → LLM 继续生成。挑战：1）多轮 tool call 的 KV cache 管理（保留中间 KV，避免重新 prefill）；2）tool 执行超时处理（设置 per-tool timeout，超时返回 error message）；3）并行 tool call（多个独立 tool 并行执行）。延迟：每轮 tool call 增加 tool_execution_time + 一次 prefill。优化：tool result 预测（speculative tool execution）。

**Q19: 如何设计 GPU 资源的 bin-packing 调度？**
**A:** 问题：不同模型需要不同 GPU 数（7B=1卡，70B=8卡，405B=16卡），如何最大化集群利用率。算法：First-Fit Decreasing（大模型优先分配）+ 碎片整理（定期迁移小模型合并空闲 GPU）。约束：TP group 必须在同一 NVLink domain（8 卡/节点），PP 可跨节点。实现：自定义 K8s scheduler（扩展 scoring plugin），考虑 GPU topology（NVLink/PCIe 连接关系）。目标：集群 GPU 利用率 >80%。

**Q20: 如何设计 LLM serving 的 graceful degradation？**
**A:** 降级策略（按优先级）：1）正常服务；2）关闭 prefix cache（释放内存给新请求）；3）降低 max_output_tokens（从 4096→1024）；4）切换到更小模型（70B→7B）；5）返回 cached response（semantic cache）；6）reject 低优先级请求（HTTP 429）。触发条件：queue_depth > threshold 或 GPU memory > 90%。实现：circuit breaker pattern，每层降级有独立阈值和恢复条件。目标：核心租户在极端负载下仍可用。

---

## Production/Debugging（20 条）

**Q1: 线上 TTFT 突然 spike 到 10s+，如何排查？**
**A:** 排查路径：1）检查 queue depth（是否请求堆积→capacity 不足）；2）检查 prefill 长度分布（是否有超长 prompt 阻塞）；3）检查 GPU memory（KV cache 是否满导致 preemption/swap）；4）检查 prefix cache hit rate（是否 cache 被 evict）；5）Nsight Systems 看是否有 kernel 异常耗时。常见原因：流量突增 + autoscaling 未及时响应、单个超长请求（100K tokens）阻塞 chunked prefill 未开启、GPU ECC error 导致降频。

**Q2: GPU OOM 在推理服务中如何处理？**
**A:** 预防：1）KV cache 预算设置为 GPU memory 的 85%（留 15% buffer）；2）per-request max_tokens 限制；3）admission control（预估 KV 需求，超出时 reject 或排队）。发生时：1）触发 preemption（swap lowest-priority 请求的 KV 到 CPU）；2）如果 swap 空间也满，recompute（丢弃 KV，后续重算）；3）最坏情况 kill 请求返回 503。监控：KV cache utilization 告警阈值 90%，OOM kill 事件计入 error budget。

**Q3: 如何诊断 GPU utilization 低的问题？**
**A:** GPU utilization 低（<30%）的常见原因：1）Batch size 太小（decode 阶段 memory-bound，单请求无法占满 GPU）→ 增大 concurrent requests；2）CPU bottleneck（tokenization、scheduling 在 CPU，GPU 等待）→ profile CPU 耗时；3）Kernel launch overhead（小 kernel 频繁 launch）→ CUDA Graphs；4）同步等待（cudaDeviceSynchronize 阻塞）→ 改为异步。工具：nvidia-smi（粗粒度）、Nsight Systems（细粒度 timeline）。

**Q4: NCCL timeout 如何排查和处理？**
**A:** NCCL timeout 表示集合通信超时（默认 5min）。排查：1）检查所有 rank 是否存活（某个 rank crash 导致其他 hang）；2）检查网络（IB/RoCE link down、交换机故障）→ `ibstat`、`nccl-tests` 验证；3）检查 GPU 状态（ECC error、XID error in dmesg）；4）检查是否有 rank 计算异常慢（straggler）。处理：设置 `NCCL_TIMEOUT`=300s + watchdog 检测 hang 后自动重启 job。预防：定期 health check 跑 allreduce benchmark。

**Q5: 推理服务的 latency regression 如何定位？**
**A:** 方法：1）对比 baseline（相同输入，新旧版本 A/B）确认是否真的退化；2）Latency breakdown（prefill_time、decode_time、scheduling_time 分别对比）；3）Kernel-level profiling（Nsight Compute 对比热点 kernel）；4）检查环境变化（driver 版本、CUDA 版本、模型权重是否变化）。常见原因：新代码引入额外 sync、batch 策略变化导致 batch size 变小、memory fragmentation 导致 allocation 变慢。回滚标准：P99 退化 >15%。

**Q6: 如何设计推理服务的 rollback 机制？**
**A:** 要求：30s 内完成回滚，不丢失 running 请求。实现：1）Blue-green deployment（新旧版本并存，流量切换）；2）旧版本保持 warm standby（模型已加载，随时接收流量）；3）回滚触发：自动（error rate >1% 持续 1min）或手动（oncall 一键）；4）流量切换通过 load balancer weight 调整（0→100% 切到旧版本）。注意：running 请求在旧实例上完成（graceful drain），新请求路由到回滚版本。

**Q7: 如何监控和优化 tail latency（P99/P999）？**
**A:** Tail latency 来源：1）长 prompt prefill（P99 用户可能发送 10x 平均长度的 prompt）；2）GC pause（Python runtime）；3）GPU frequency throttling（温度过高降频）；4）网络 jitter（跨节点通信）。优化：1）Chunked prefill 限制单次 prefill 时间；2）请求级 timeout + 重试（不同节点）；3）over-provision 20%（减少排队）；4）isolate heavy requests 到专用 pool。目标：P99/P50 < 3x。

**Q8: 推理服务出现间歇性错误输出（乱码/重复），如何排查？**
**A:** 可能原因：1）GPU 硬件错误（ECC uncorrectable error）→ 检查 `nvidia-smi -q` 的 ECC errors 计数；2）KV cache corruption（内存管理 bug，block 被错误复用）→ 对比相同输入多次输出是否一致；3）量化精度问题（某些 layer 对量化敏感）→ 用 FP16 baseline 对比；4）并发 bug（race condition in scheduler）→ 单请求测试是否正常。诊断：记录 problematic request 的完整 context，replay 验证是否可复现。

**Q9: 如何处理推理服务的 memory leak？**
**A:** 症状：GPU memory 随时间缓慢增长，最终 OOM。排查：1）监控 `torch.cuda.memory_allocated()` 趋势；2）检查 KV cache block 是否正确释放（block manager 的 free list 是否增长）；3）检查 Python 对象引用（`gc.get_referrers()`）；4）CUDA memory 工具（`cuda-memcheck`、`compute-sanitizer`）。常见原因：异常请求未正确清理 KV cache、Python 循环引用、CUDA event 未释放。修复后验证：跑 24h stress test 确认 memory 稳定。

**Q10: 如何做推理服务的 load testing？**
**A:** 工具：自定义 benchmark client（模拟真实 input/output 分布）。关键参数：1）QPS ramp（从 0 逐步增加到 saturation）；2）Input length 分布（从 access log 采样）；3）Output length 分布；4）并发连接数。指标：throughput（tokens/s）、TTFT P50/P99、TPOT P50/P99、error rate。Saturation point：TTFT P99 > SLA 时的 QPS。容量规划：production QPS = saturation * 0.7（30% headroom）。

**Q11: GPU 降频（thermal throttling）如何检测和处理？**
**A:** 检测：`nvidia-smi -q -d CLOCK` 看实际频率 vs 最大频率，`nvidia-smi dmon` 持续监控温度和功耗。A100 在 >83°C 开始降频，>90°C 严重降频（性能降 20-30%）。处理：1）检查散热（风扇转速、机房温度）；2）降低 GPU 负载（减少 batch size）；3）设置 power limit（`nvidia-smi -pl 300` 限制功耗换取温度稳定）。预防：机房温度监控 + 告警，GPU 间留足散热空间。

**Q12: 如何排查推理服务的 CPU bottleneck？**
**A:** 症状：GPU utilization 低但 throughput 上不去。排查：1）`top`/`htop` 看 CPU 使用率（tokenizer、scheduler 是否打满 CPU）；2）Python profiler（cProfile）看热点函数；3）检查 GIL 竞争（多线程 Python 代码）。常见瓶颈：tokenization（大 vocab 的 BPE 编码慢）→ 用 Rust tokenizer（HuggingFace tokenizers）；scheduling 逻辑复杂 → 优化数据结构（O(n)→O(logn)）；序列化/反序列化 → 用 protobuf 替代 JSON。

**Q13: 推理服务如何处理 poison request（导致 crash 的请求）？**
**A:** 检测：同一请求重试 3 次都 crash → 标记为 poison。处理：1）隔离到 dead letter queue；2）返回用户 500 + 错误信息；3）记录完整 request payload 供后续分析。预防：1）input validation（max_tokens 限制、非法字符过滤）；2）sandbox execution（每个请求在独立进程/容器，crash 不影响其他请求）；3）watchdog timer（单请求超时 5min 强制 kill）。根因分析：replay poison request 在 debug 环境复现。

**Q14: 如何设计推理服务的 SLA 和 error budget？**
**A:** SLA 定义：可用性 99.9%（月 downtime <43min）、TTFT P99 <2s、TPOT P50 <50ms、error rate <0.1%。Error budget = 1 - SLA = 0.1%（每月允许 43min 不可用或 0.1% 请求失败）。消耗 error budget 时：freeze deployment、focus on reliability。监控：实时 error budget burn rate（如果 1h 内消耗了 10% monthly budget → 告警）。SLI（Service Level Indicator）：从 client 侧测量（包含网络延迟），不是 server 侧。

**Q15: 如何排查推理结果质量突然下降？**
**A:** 排查路径：1）检查模型版本是否变更（意外部署了错误权重）；2）检查 prompt template 是否变更（配置错误）；3）检查 quantization 是否异常（calibration 数据问题）；4）检查 KV cache 是否 corruption（硬件错误）；5）检查 sampling 参数是否变更（temperature、top_p）。验证：用 golden test set 跑 eval，对比 baseline。快速定位：git blame 最近变更 + 二分法回滚确认引入点。

**Q16: 多 GPU 推理中某张卡明显慢（straggler），如何处理？**
**A:** 检测：per-GPU kernel 执行时间监控，某卡持续 >1.5x 平均值。原因：1）ECC error 导致降频；2）PCIe 带宽退化（link degradation）；3）thermal throttling（散热不均）；4）后台进程占用（其他 job 未清理）。处理：1）短期：drain 该节点，流量切到其他节点；2）中期：替换故障 GPU；3）长期：定期 health check（跑 GEMM benchmark，性能 <90% baseline 标记为 unhealthy）。TP 场景 straggler 影响整个 group。

**Q17: 如何做推理服务的 chaos engineering？**
**A:** 实验设计：1）Kill random GPU worker（验证 failover <30s）；2）注入网络延迟（验证 timeout + retry 机制）；3）模拟 GPU OOM（验证 preemption 和 graceful degradation）；4）模拟流量 spike 10x（验证 autoscaling + rate limiting）。执行：先在 staging 环境，确认 runbook 有效后在 production 低峰期执行。工具：Chaos Monkey（kill instance）、tc（网络延迟）、自定义 fault injection。前提：完善的 observability + 自动回滚能力。

**Q18: 推理服务的日志和 tracing 最佳实践？**
**A:** 日志：structured JSON（request_id、model、input_tokens、output_tokens、ttft、tpot、status）。采样：100% error log、10% success log（高 QPS 时）。Tracing：OpenTelemetry span 覆盖完整链路（API gateway → router → scheduler → prefill → decode → response）。关键 span 属性：queue_wait_time、prefill_time、decode_time、total_tokens。存储：日志 30 天（S3 archive），traces 7 天（Jaeger/Tempo）。查询：按 request_id 关联 log + trace + metrics。

**Q19: 如何处理推理服务的 thundering herd 问题？**
**A:** 场景：大量 client 同时重试（如 server 重启后）或 cache 失效导致请求洪峰。处理：1）Exponential backoff + jitter（client 重试间隔随机化）；2）Rate limiting（per-client + global）；3）Circuit breaker（错误率 >50% 时 fast-fail，不发送请求）；4）Warm-up period（重启后逐步接收流量，不立即 full load）。预防：graceful shutdown（先停止接收新请求，drain 完成后再关闭）避免同时重启。

**Q20: 如何评估和改善推理服务的可靠性？**
**A:** 评估：1）MTBF（Mean Time Between Failures）：目标 >720h（月级）；2）MTTR（Mean Time To Recovery）：目标 <5min；3）Error budget 消耗率。改善：1）消除 SPOF（单点故障）：多副本 + 跨 AZ 部署；2）自动化恢复：health check + auto-restart + auto-scaling；3）定期演练：每月 chaos engineering + 季度 DR drill；4）Post-mortem：每次 incident 后 root cause analysis + action items。目标：从 99.9% 提升到 99.95%（月 downtime 从 43min 降到 22min）。
