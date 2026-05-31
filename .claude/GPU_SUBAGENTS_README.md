# Claude Code GPU Learning Subagents

已创建以下 subagents：

1. cuda-kernel-writer
2. inference-framework-writer
3. post-training-writer
4. gpu-profiler-writer
5. curriculum-reviewer

## 使用方式

在项目根目录运行：

```bash
claude
```

进入 Claude Code 后，粘贴：

```text
请并行调用 5 个 subagents，生成《GPU 推理系统、CUDA 算子、SFT/RL 后训练与性能工程学习材料全集》。

Subagents：
1. @agent-cuda-kernel-writer
2. @agent-inference-framework-writer
3. @agent-post-training-writer
4. @agent-gpu-profiler-writer
5. @agent-curriculum-reviewer

并行任务：

1. @agent-cuda-kernel-writer
   生成《CUDA / Triton 推理算子学习材料》。
   覆盖：
   - CUDA execution model
   - memory hierarchy
   - global memory coalescing
   - shared memory bank conflict
   - register pressure
   - occupancy
   - RMSNorm
   - RoPE
   - Softmax
   - GEMM
   - Attention
   - FlashAttention
   - PagedAttention
   - KV Cache
   - MoE

2. @agent-inference-framework-writer
   生成《vLLM / SGLang / TensorRT-LLM / llama.cpp 推理框架调优材料》。
   覆盖：
   - prefill
   - decode
   - continuous batching
   - PagedAttention
   - KV cache
   - prefix caching
   - chunked prefill
   - speculative decoding
   - tensor parallel
   - pipeline parallel
   - data parallel serving
   - TTFT
   - TPOT
   - throughput
   - latency
   - QPS
   - 显存权衡
   - serving benchmark

3. @agent-post-training-writer
   生成《SFT / RL 后训练工程材料》。
   覆盖：
   - SFT
   - LoRA
   - QLoRA
   - DPO
   - IPO
   - KTO
   - ORPO
   - Reward Model
   - PPO
   - GRPO
   - RLVR
   - Agentic RL
   - Rollout Engine
   - vLLM / SGLang rollout backend
   - KL penalty
   - reward hacking
   - training throughput tuning

4. @agent-gpu-profiler-writer
   生成《GPU profiling 与性能调优材料》。
   覆盖：
   - Nsight Systems
   - Nsight Compute
   - PyTorch Profiler
   - CUDA event timing
   - roofline analysis
   - occupancy analysis
   - memory bandwidth analysis
   - warp stall analysis
   - Tensor Core utilization
   - L2 cache
   - DRAM throughput
   - NCCL profiling
   - host bottleneck
   - synchronization bottleneck
   - performance regression diagnosis

5. @agent-curriculum-reviewer
   在前 4 个 subagents 返回结果后执行审查。
   审查：
   - 概念完整性
   - 数学定义准确性
   - 性能分析可验证性
   - profiling 指标覆盖度
   - benchmark 设计合理性
   - 实验可操作性
   - 习题难度分布
   - 工程案例具体性
   - 跨章节一致性

执行要求：
- 前 4 个 subagents 并行执行。
- 每个 subagent 输出 Markdown。
- 每个 subagent 输出目录、正文模板、实验模板、题库模板、benchmark 表格、profiling 解读模板。
- 不要让中间日志进入主会话。
- 每个 subagent 只返回结构化最终结果。
- 主会话收到前 4 个结果后，再调用 curriculum-reviewer。
- 最后由主会话合并成统一的 8 卷学习材料目录。

最终输出：
1. 8 卷总目录
2. 每卷章节表
3. 每章固定模板
4. 每类实验模板
5. 每类题库模板
6. benchmark 表格模板
7. profiling 解读模板
8. 后续逐章生成的调用方式
```

也可以直接输入：

```text
请并行调用 @agent-cuda-kernel-writer、@agent-inference-framework-writer、@agent-post-training-writer、@agent-gpu-profiler-writer 生成 GPU 推理系统学习材料，完成后调用 @agent-curriculum-reviewer 审查并合并。
```

## 检查 subagents

在 Claude Code 内运行：

```text
/agents
```

## 注意

如果当前已经打开 Claude Code session，修改 subagent 文件后需要重启 Claude Code session。
