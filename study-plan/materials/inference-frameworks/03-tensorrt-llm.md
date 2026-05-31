# 03 - TensorRT-LLM 推理优化

## 1. 学习目标

- 理解 TensorRT-LLM 的架构设计与编译流程
- 掌握从 HuggingFace 模型到 TensorRT 引擎的转换过程
- 学会配置 TensorRT-LLM 的关键参数以优化推理性能
- 理解 TensorRT-LLM 中的算子融合（Operator Fusion）与内核自动调优（Kernel Auto-Tuning）机制
- 能够在生产环境中部署并监控 TensorRT-LLM 服务

## 2. 系统动机

大语言模型推理面临的核心挑战是：模型参数量巨大导致计算密集，自回归解码（Autoregressive Decoding）导致内存带宽受限。通用推理框架（如 PyTorch）虽然灵活，但无法充分利用 GPU 硬件特性。

TensorRT-LLM 由 NVIDIA 开发，专门针对 LLM 推理场景优化，核心动机包括：

- **编译期优化**：将动态图转为静态图，消除 Python 解释器开销
- **算子融合**：将多个小算子合并为单个高效 CUDA 内核
- **量化支持**：原生支持 FP8/INT8/INT4 量化，最大化 Tensor Core 利用率
- **内存优化**：通过 KV Cache 复用、权重流式加载减少显存占用
- **多 GPU 支持**：内置张量并行（Tensor Parallelism）与流水线并行（Pipeline Parallelism）

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| TRT Engine | TensorRT Engine | 编译后的优化推理引擎二进制文件 |
| Builder | TensorRT Builder | 将网络定义转换为优化引擎的编译器 |
| Plugin | TensorRT Plugin | 自定义算子实现，扩展 TensorRT 原生能力 |
| IFB | In-Flight Batching | 动态批处理，允许请求随时加入/退出批次 |
| MHA | Multi-Head Attention | 多头注意力机制 |
| GQA | Grouped Query Attention | 分组查询注意力，减少 KV 头数量 |
| MQA | Multi-Query Attention | 多查询注意力，所有头共享 KV |
| Weight Streaming | Weight Streaming | 权重流式加载，支持超出显存的模型 |
| Strongly Typed | Strongly Typed Network | 精确控制每层数据类型的网络定义方式 |
| FMHA | Fused Multi-Head Attention | 融合多头注意力内核 |

## 4. 执行流程

### 4.1 模型转换流程

```
HuggingFace 模型
    │
    ▼
convert_checkpoint.py（权重格式转换）
    │
    ▼
TensorRT-LLM Model Definition（Python 网络定义）
    │
    ▼
trtllm-build（编译构建）
    │
    ▼
TensorRT Engine（.engine 文件）
    │
    ▼
Triton Inference Server / C++ Runtime 部署
```

### 4.2 推理执行流程

```
1. 接收请求 → Tokenize
2. 调度器决定批次组成（IFB）
3. Prefill 阶段：并行处理输入 tokens
4. Decode 阶段：逐 token 生成
   - KV Cache 分配与管理
   - 采样策略执行（Top-K/Top-P/Beam Search）
5. 检测停止条件 → Detokenize → 返回结果
```

### 4.3 编译优化阶段

```
Layer Fusion → Kernel Selection → Memory Planning → Precision Calibration → Engine Serialization
```

## 5. 参数解释

### 5.1 构建参数（trtllm-build）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_batch_size` | 8 | 引擎支持的最大批次大小 |
| `--max_input_len` | 1024 | 最大输入序列长度 |
| `--max_seq_len` | 2048 | 最大总序列长度（输入+输出） |
| `--max_num_tokens` | - | 单次迭代最大 token 数 |
| `--gemm_plugin` | auto | GEMM 算子实现选择 |
| `--gpt_attention_plugin` | auto | 注意力算子实现 |
| `--context_fmha` | enabled | 启用融合上下文注意力 |
| `--paged_kv_cache` | enabled | 启用分页 KV Cache |
| `--tokens_per_block` | 64 | 每个 KV Cache 块的 token 数 |
| `--use_custom_all_reduce` | enabled | 自定义 AllReduce 通信 |
| `--multi_block_mode` | disabled | 长序列多块注意力模式 |

### 5.2 运行时参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_tokens_in_paged_kv_cache` | - | KV Cache 最大 token 容量 |
| `kv_cache_free_gpu_mem_fraction` | 0.9 | KV Cache 占用空闲显存比例 |
| `enable_chunked_context` | false | 分块处理长上下文 |
| `batch_scheduler_policy` | guaranteed_no_evict | 调度策略 |
| `decoding_mode` | auto | 解码模式选择 |

### 5.3 量化参数

| 参数 | 说明 |
|------|------|
| `--use_fp8` | 启用 FP8 量化（Hopper 架构） |
| `--int8_kv_cache` | KV Cache 使用 INT8 |
| `--weight_only_precision` | 仅权重量化精度（int8/int4） |
| `--smoothquant` | SmoothQuant 量化系数 |

## 6. 调优目标

| 目标 | 指标 | 典型基线 | 优化目标 |
|------|------|----------|----------|
| 首 Token 延迟 | TTFT (Time To First Token) | <500ms | <200ms |
| 生成吞吐 | Tokens/s/GPU | 1000-2000 | 3000-5000 |
| 端到端延迟 | E2E Latency P99 | <5s | <2s |
| 显存利用率 | GPU Memory Usage | 60-70% | 85-95% |
| 批次利用率 | Batch Utilization | 50% | >80% |

## 7. 适用场景

- **高吞吐在线服务**：需要最大化单 GPU 吞吐量的 API 服务
- **延迟敏感场景**：对 TTFT 和 ITL（Inter-Token Latency）有严格要求
- **NVIDIA GPU 专用部署**：A100/H100/H200 等数据中心 GPU
- **固定模型长期服务**：模型不频繁更新，可接受编译时间
- **多 GPU 大模型部署**：70B+ 参数模型的张量并行部署
- **量化部署**：需要 FP8/INT8 量化以降低成本

## 8. 不适用场景

- **快速原型验证**：编译时间长（数十分钟到数小时），不适合频繁实验
- **非 NVIDIA 硬件**：仅支持 NVIDIA GPU，不支持 AMD/Intel
- **动态模型结构**：模型结构频繁变化时编译成本过高
- **极小批次低延迟**：batch_size=1 时优势不明显，vLLM 可能更灵活
- **资源受限环境**：编译过程本身需要大量 CPU 内存和时间
- **需要模型热更新**：引擎文件与模型绑定，更新需重新编译

## 9. 副作用

- **编译时间长**：大模型编译可能需要 1-4 小时
- **引擎文件体积大**：编译后的 .engine 文件可能比原始权重更大
- **精度损失**：FP16/FP8 编译可能引入微小数值差异
- **调试困难**：编译后的引擎是黑盒，难以定位具体层的问题
- **版本耦合**：引擎与 TensorRT 版本、GPU 架构强绑定
- **内存碎片**：长时间运行后 KV Cache 可能产生碎片

## 10. 风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 编译参数不匹配 | 运行时 OOM 或性能退化 | 基于实际负载 profile 设置参数 |
| 版本升级不兼容 | 引擎需要重新编译 | 维护编译脚本，CI/CD 自动化 |
| 量化精度损失 | 模型输出质量下降 | 量化前后对比评估，校准数据集 |
| KV Cache 耗尽 | 请求被拒绝或排队 | 监控 KV Cache 使用率，设置合理上限 |
| GPU 显存泄漏 | 服务逐渐退化 | 定期重启，监控显存趋势 |
| 单点故障 | 服务不可用 | 多副本部署，健康检查 |

## 11. 验证方式

### 11.1 正确性验证

```bash
# 对比 HuggingFace 原始输出与 TRT-LLM 输出
python compare_outputs.py \
  --hf_model_dir /models/llama-7b \
  --engine_dir /engines/llama-7b-fp16 \
  --test_prompts test_prompts.jsonl \
  --tolerance 1e-3
```

### 11.2 性能验证

```bash
# 使用官方 benchmark 工具
python benchmarks/benchmark_throughput.py \
  --engine_dir /engines/llama-7b-fp16 \
  --dataset ShareGPT \
  --num_requests 1000 \
  --concurrency 64
```

### 11.3 量化质量验证

```bash
# 评估量化后模型在标准数据集上的表现
python eval.py \
  --engine_dir /engines/llama-7b-int8 \
  --eval_task mmlu,hellaswag \
  --num_fewshot 5
```

## 12. 监控指标

| 指标 | 采集方式 | 告警阈值 |
|------|----------|----------|
| `trtllm_request_throughput` | Triton Metrics | <预期 80% |
| `trtllm_kv_cache_utilization` | Runtime Stats | >95% |
| `trtllm_inflight_batching_requests` | Runtime Stats | 持续满载 |
| `trtllm_time_to_first_token_ms` | 请求日志 | P99 >500ms |
| `trtllm_inter_token_latency_ms` | 请求日志 | P99 >100ms |
| `gpu_memory_used_bytes` | DCGM/nvidia-smi | >95% 总显存 |
| `gpu_utilization_percent` | DCGM | <50%（可能瓶颈在别处） |
| `request_queue_depth` | 调度器 | 持续增长 |

## 13. 压测方法

### 13.1 基准吞吐测试

```bash
# 固定输入输出长度，测试最大吞吐
python benchmark_throughput.py \
  --input_len 512 --output_len 256 \
  --num_requests 5000 \
  --concurrency 128
```

### 13.2 真实负载模拟

```bash
# 使用 ShareGPT 数据集模拟真实分布
python benchmark_serving.py \
  --backend trtllm \
  --endpoint http://localhost:8000/v2/models/llama/generate \
  --dataset-name sharegpt \
  --request-rate 10 \
  --num-prompts 1000
```

### 13.3 极限压力测试

```bash
# 逐步增加并发直到服务降级
for rate in 10 20 50 100 200; do
  echo "Testing rate: $rate req/s"
  python benchmark_serving.py --request-rate $rate --duration 300
  sleep 60  # 冷却期
done
```

### 13.4 长序列压测

```bash
# 测试长上下文场景
python benchmark_throughput.py \
  --input_len 8192 --output_len 2048 \
  --num_requests 100 \
  --concurrency 16
```

## 14. Profiling 方法

### 14.1 Nsight Systems 全局分析

```bash
nsys profile -o trtllm_profile \
  --trace=cuda,nvtx \
  python run_inference.py --engine_dir /engines/llama-7b
```

### 14.2 Nsight Compute 内核分析

```bash
ncu --set full \
  --target-processes all \
  -o kernel_profile \
  python run_inference.py --engine_dir /engines/llama-7b --num_requests 1
```

### 14.3 TensorRT Profiler

```python
# 在代码中启用逐层 profiling
import tensorrt as trt
context.profiler = trt.Profiler()
# 执行推理后查看各层耗时
```

### 14.4 内存分析

```bash
# 监控显存分配模式
nvidia-smi dmon -s mu -d 1 -f gpu_mem_trace.csv &
python run_inference.py --engine_dir /engines/llama-7b
```

## 15. 失败案例

### 案例 1：max_batch_size 设置过大导致 OOM

**现象**：服务启动后处理少量请求即 OOM 崩溃。

**原因**：`max_batch_size=256` 编译时预分配了过多 KV Cache 空间，实际显存不足。

**修复**：根据公式计算合理值：
```
KV Cache 显存 = max_batch_size × max_seq_len × num_layers × 2 × num_kv_heads × head_dim × dtype_size
```

### 案例 2：未启用 context_fmha 导致 Prefill 慢

**现象**：TTFT 比预期高 3-5 倍。

**原因**：未启用 `--context_fmha enable`，Prefill 阶段使用了未融合的注意力实现。

**修复**：重新编译时添加 `--context_fmha enable`。

### 案例 3：量化校准数据不匹配

**现象**：INT8 量化后模型输出质量严重下降。

**原因**：校准数据集与实际使用场景差异大，导致量化范围不准确。

**修复**：使用与生产数据分布相似的校准集，增加校准样本数量。

### 案例 4：多 GPU 通信瓶颈

**现象**：2 GPU 张量并行性能仅为单 GPU 的 1.3 倍。

**原因**：使用 PCIe 连接而非 NVLink，AllReduce 通信成为瓶颈。

**修复**：确认 GPU 拓扑，使用 NVLink 连接的 GPU 对；或改用流水线并行。

## 16. 复盘模板

```markdown
## TensorRT-LLM 调优复盘

### 基本信息
- 日期：
- 模型：
- GPU 型号/数量：
- TensorRT-LLM 版本：

### 目标
- 优化指标：
- 基线值：
- 目标值：

### 编译配置
- max_batch_size：
- max_input_len：
- max_seq_len：
- 量化方式：
- 插件配置：

### 实验过程
| 实验 | 变更 | TTFT | 吞吐 | 显存 | 结论 |
|------|------|------|------|------|------|
| baseline | - | | | | |
| exp1 | | | | | |

### 关键发现
-

### 遗留问题
-

### 后续计划
-
```

## 17. 实验任务

### 实验 1：基础编译与部署

将 Llama-2-7B 从 HuggingFace 格式转换为 TensorRT-LLM 引擎，使用 FP16 精度，部署到 Triton Inference Server。

### 实验 2：量化对比

分别编译 FP16、INT8 Weight-Only、FP8（如有 H100）版本，对比吞吐、延迟和输出质量。

### 实验 3：参数调优

固定模型和硬件，调整 `max_batch_size`、`tokens_per_block`、`max_num_tokens` 参数，找到最优配置。

### 实验 4：多 GPU 并行

使用 2/4 GPU 进行张量并行部署，测量扩展效率，分析通信开销。

### 实验 5：长序列优化

启用 `multi_block_mode` 和 `enable_chunked_context`，测试 8K/16K/32K 输入长度的性能表现。

## 18. 习题

1. TensorRT-LLM 的编译过程主要包含哪些优化阶段？各阶段的作用是什么？
2. 解释 `context_fmha` 的作用，为什么它对 Prefill 性能至关重要？
3. `max_batch_size` 设置过大会带来什么问题？如何计算合理值？
4. In-Flight Batching 与 Static Batching 的核心区别是什么？
5. 为什么 TensorRT-LLM 引擎文件与 GPU 架构绑定？能否跨架构使用？
6. 解释 `tokens_per_block` 参数对 KV Cache 内存效率的影响。
7. FP8 量化相比 INT8 量化有什么优势？适用于哪些 GPU 架构？
8. 如何判断推理瓶颈是 Compute Bound 还是 Memory Bound？
9. `use_custom_all_reduce` 在什么场景下应该禁用？
10. 解释 Weight Streaming 功能的工作原理和适用场景。
11. 为什么 TensorRT-LLM 需要在编译时指定 `max_input_len` 和 `max_seq_len`？
12. 如何在不重新编译引擎的情况下调整运行时行为？
13. Paged KV Cache 在 TensorRT-LLM 中的实现与 vLLM 有何异同？
14. 解释 `batch_scheduler_policy` 的 `guaranteed_no_evict` 和 `max_utilization` 策略的区别。
15. 多 GPU 张量并行时，AllReduce 通信量与模型哪些维度相关？
16. 如何使用 Nsight Systems 定位 TensorRT-LLM 推理中的性能瓶颈？
17. 编译时启用 `--remove_input_padding` 的作用是什么？
18. 如何评估量化后模型的输出质量？常用哪些评估指标？
19. TensorRT-LLM 的 Plugin 机制解决了什么问题？举例说明。
20. 在生产环境中，如何实现 TensorRT-LLM 引擎的无缝升级？

## 19. 标准答案

1. **编译优化阶段**：(1) 图优化：常量折叠、死代码消除；(2) 层融合：合并相邻算子；(3) 内核选择：为每层选择最优 CUDA 内核；(4) 内存规划：优化中间张量的内存分配；(5) 精度校准：确定量化参数。

2. **context_fmha**：将 Q/K/V 投影、注意力计算、Softmax、输出投影融合为单个内核。避免了中间结果写回全局内存的开销，对 Prefill 阶段（大量 token 并行计算注意力）性能提升显著，通常可提升 2-4 倍。

3. **max_batch_size 过大**：预分配 KV Cache 占用过多显存，可能导致 OOM 或实际可用 KV Cache 容量不足。计算公式：`KV_mem = batch × seq_len × layers × 2 × kv_heads × head_dim × dtype_bytes`。应根据实际并发量和显存预算反推。

4. **IFB vs Static Batching**：Static Batching 要求批次内所有请求同时开始、同时结束，短请求需等待长请求。IFB 允许请求随时加入和退出批次，已完成的请求立即释放资源，新请求可立即填入，显著提高 GPU 利用率。

5. **架构绑定原因**：编译时针对特定 GPU 的 SM 数量、共享内存大小、Tensor Core 版本选择最优内核和执行计划。不同架构（Ampere vs Hopper）的硬件特性不同，优化策略不可互换。

6. **tokens_per_block**：值越小，内存分配粒度越细，浪费越少但管理开销越大；值越大，分配效率高但可能浪费空间。通常 64-128 是较好的平衡点。

7. **FP8 vs INT8**：FP8 保留浮点表示的动态范围，对异常值更鲁棒，不需要复杂的校准过程。FP8 仅支持 Hopper（H100）及更新架构，INT8 支持 Ampere 及以上。

8. **判断瓶颈**：使用 Nsight Compute 分析内核的 Arithmetic Intensity。若 SM 利用率高但内存带宽未饱和则为 Compute Bound；若内存带宽接近峰值但 SM 利用率低则为 Memory Bound。Decode 阶段通常是 Memory Bound。

9. **禁用 custom_all_reduce**：当 GPU 间通过 PCIe 连接（非 NVLink）时，自定义 AllReduce 可能不如 NCCL 默认实现高效；或在某些特殊拓扑下出现正确性问题时应禁用。

10. **Weight Streaming**：将模型权重存储在 CPU 内存中，推理时按需流式传输到 GPU。适用于模型权重超出单 GPU 显存但不想使用多 GPU 的场景，以带宽换显存。

11. **编译时指定长度**：TensorRT 需要在编译时确定张量形状范围以优化内存分配和内核选择。运行时超出编译时指定范围的请求将被拒绝。

12. **运行时调整**：可调整 `kv_cache_free_gpu_mem_fraction`、调度策略、采样参数、并发限制等。这些参数不影响引擎内部计算图，仅影响外围调度和资源管理。

13. **Paged KV Cache 异同**：相同点是都借鉴操作系统虚拟内存分页思想，按块分配 KV Cache。不同点是 TensorRT-LLM 的实现与编译引擎深度集成，块大小在编译时确定；vLLM 的实现更灵活，可运行时动态调整。

14. **调度策略区别**：`guaranteed_no_evict` 保证已接受的请求不会被驱逐，可能导致新请求排队；`max_utilization` 允许过度承诺，在内存不足时驱逐低优先级请求，吞吐更高但延迟波动更大。

15. **AllReduce 通信量**：与隐藏层维度（hidden_size）和批次中的 token 数相关。每次 AllReduce 传输 `batch_tokens × hidden_size × dtype_bytes` 数据。与层数无关（每层独立通信）。

16. **Nsight Systems 定位瓶颈**：查看 CUDA 内核时间线，识别长耗时内核和 GPU 空闲间隙。关注 NVTX 标记区分 Prefill/Decode 阶段，检查 Host-Device 同步点和内存拷贝。

17. **remove_input_padding**：移除批次内不同长度序列的 padding token，将所有有效 token 紧密排列。减少无效计算，对批次内序列长度差异大的场景提升明显。

18. **量化质量评估**：使用 Perplexity 评估语言建模能力；使用 MMLU、HellaSwag 等基准测试评估任务能力；使用人工评估或 LLM-as-Judge 评估生成质量。关注与 FP16 基线的差距。

19. **Plugin 机制**：解决 TensorRT 原生不支持的自定义算子问题。例如 RoPE（旋转位置编码）、FlashAttention 变体、自定义激活函数等，通过 Plugin 以 CUDA 代码实现并注册到 TensorRT。

20. **无缝升级**：采用蓝绿部署或金丝雀发布。新引擎编译完成后部署到新实例，通过负载均衡器逐步切换流量。保留旧引擎实例作为回滚目标，验证新引擎的正确性和性能后完全切换。

## 20. 调优 Checklist

- [ ] 确认 GPU 型号和 TensorRT-LLM 版本兼容性
- [ ] 分析实际负载的输入/输出长度分布
- [ ] 根据负载分布设置 `max_input_len` 和 `max_seq_len`
- [ ] 计算 KV Cache 显存需求，确定 `max_batch_size`
- [ ] 启用 `--context_fmha enable` 优化 Prefill
- [ ] 启用 `--paged_kv_cache enable` 优化内存利用
- [ ] 启用 `--remove_input_padding` 减少无效计算
- [ ] 选择合适的量化策略（FP16/FP8/INT8）
- [ ] 如使用量化，准备代表性校准数据集
- [ ] 验证量化后模型输出质量
- [ ] 配置 `kv_cache_free_gpu_mem_fraction` 最大化缓存
- [ ] 选择合适的 `batch_scheduler_policy`
- [ ] 多 GPU 场景确认 NVLink 拓扑
- [ ] 运行基准测试确认性能达标
- [ ] 进行长时间稳定性测试（>24h）
- [ ] 配置监控告警（TTFT、吞吐、显存、队列深度）
- [ ] 准备回滚方案和引擎版本管理
- [ ] 文档化编译参数和部署配置
