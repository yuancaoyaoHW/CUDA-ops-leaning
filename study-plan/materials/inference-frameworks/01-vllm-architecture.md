# vLLM 整体架构

## 1. 学习目标

- 理解 vLLM 的端到端请求处理流程，从 API 接收到 token 输出
- 掌握 vLLM 核心组件（Scheduler、BlockManager、Worker、ModelRunner）之间的协作关系
- 能够根据业务场景选择合适的 vLLM 配置参数并进行性能调优
- 理解 vLLM 如何通过 PagedAttention 实现高效显存管理
- 掌握 vLLM 在不同硬件配置下的部署策略
- 能够诊断 vLLM 常见性能瓶颈并给出解决方案

## 2. 系统动机

### 2.1 传统推理框架的问题

传统 LLM 推理框架（如原生 HuggingFace Transformers）存在以下核心问题：

1. **显存碎片化**：KV cache 按最大序列长度预分配，实际使用率低于 50%
2. **批处理效率低**：静态批处理（Static Batching）要求同一批次所有请求同时完成，短请求被长请求阻塞
3. **吞吐量受限**：无法充分利用 GPU 计算能力，decode 阶段 GPU 利用率通常低于 30%
4. **缺乏生产级特性**：无内置的请求调度、优先级管理、流式输出等能力

### 2.2 vLLM 的设计目标

vLLM（Virtual Large Language Model）由 UC Berkeley 团队开发，核心设计目标：

- **最大化吞吐量**：通过 continuous batching + PagedAttention 实现接近理论上限的吞吐
- **最小化显存浪费**：将 KV cache 显存利用率从 ~50% 提升到 >95%
- **生产就绪**：提供 OpenAI 兼容 API、流式输出、多模型管理等生产特性
- **易于扩展**：支持 tensor parallel、pipeline parallel 等分布式推理

### 2.3 性能指标权衡

| 指标 | 定义 | vLLM 优化方向 |
|------|------|---------------|
| TTFT (Time To First Token) | 首 token 延迟 | 通过 chunked prefill 控制 |
| TPOT (Time Per Output Token) | 每 token 生成延迟 | 通过批处理摊薄 |
| Throughput | 单位时间生成 token 数 | 核心优化目标 |
| QPS (Queries Per Second) | 每秒处理请求数 | 受 batch size 和序列长度影响 |
| 显存占用 | GPU memory usage | PagedAttention 优化 |
| 并发数 | 同时处理的请求数 | 受显存和调度策略限制 |

## 3. 核心术语表

| 术语 | 英文全称 | 含义 |
|------|----------|------|
| PagedAttention | Paged Attention | 将 KV cache 分页管理的注意力机制 |
| Block | Physical/Logical Block | KV cache 的最小分配单元 |
| BlockManager | Block Manager | 管理物理块与逻辑块映射的组件 |
| Scheduler | Request Scheduler | 决定哪些请求进入当前批次的调度器 |
| SequenceGroup | Sequence Group | 同一请求的多个候选序列（beam search） |
| Worker | Model Worker | 执行模型前向推理的工作进程 |
| ModelRunner | Model Runner | 封装模型执行逻辑的组件 |
| Engine | LLM Engine | vLLM 核心引擎，协调调度与执行 |
| AsyncEngine | Async LLM Engine | 异步引擎，支持并发请求处理 |
| SamplingParams | Sampling Parameters | 采样参数（temperature、top_p 等） |
| Prefix Caching | Prefix Caching | 共享前缀的 KV cache 复用机制 |
| Chunked Prefill | Chunked Prefill | 将长 prefill 分块执行以降低 TTFT |
| CUDA Graph | CUDA Graph | 预录制 GPU 操作图以减少 kernel launch 开销 |

## 4. 执行流程

### 4.1 请求生命周期

```
Client Request (OpenAI API)
    │
    ▼
AsyncLLMEngine.generate()
    │
    ▼
Scheduler.schedule()
    ├── 检查可用物理块
    ├── 决定 prefill / decode 请求
    ├── 处理 preemption（抢占）
    │
    ▼
Worker.execute_model()
    ├── 准备输入张量
    ├── 执行模型前向传播
    ├── 采样输出 token
    │
    ▼
Scheduler.update()
    ├── 更新序列状态
    ├── 释放已完成序列的块
    │
    ▼
Stream Response to Client
```

### 4.2 Scheduler 调度逻辑

Scheduler 每个 step 执行以下决策：

1. **Waiting 队列**：新到达的请求，等待 prefill
2. **Running 队列**：正在 decode 的请求
3. **Swapped 队列**：被换出到 CPU 的请求

调度优先级：
- 优先处理 Running 队列（避免已投入资源的请求被饿死）
- 其次处理 Swapped 队列（恢复被抢占的请求）
- 最后处理 Waiting 队列（接纳新请求）

### 4.3 BlockManager 块管理

```python
# 逻辑块到物理块的映射
logical_block_0 -> physical_block_42
logical_block_1 -> physical_block_17
logical_block_2 -> physical_block_89

# Copy-on-Write (CoW) 机制
# 当 beam search 分叉时，共享已有块，仅在写入时复制
sequence_1.block_0 -> physical_block_42  # 共享
sequence_2.block_0 -> physical_block_42  # 共享
sequence_1.block_1 -> physical_block_17  # 独占（已写入新 token）
```

### 4.4 Worker 执行流程

```python
class Worker:
    def execute_model(self, seq_group_metadata_list):
        # 1. 准备输入
        input_tokens, input_positions, block_tables = self.prepare_input(...)
        
        # 2. 执行模型
        hidden_states = self.model(input_tokens, input_positions, 
                                    kv_caches, block_tables)
        
        # 3. 采样
        output = self.sampler(hidden_states, sampling_params)
        return output
```

## 5. 参数解释

### 5.1 引擎级参数

| 参数 | 默认值 | 含义 | 调优建议 |
|------|--------|------|----------|
| `--model` | 必填 | 模型路径或 HuggingFace ID | - |
| `--tensor-parallel-size` | 1 | TP 并行度 | 模型放不下单卡时增加 |
| `--pipeline-parallel-size` | 1 | PP 并行度 | 需要更多卡但 TP 通信瓶颈时使用 |
| `--max-model-len` | 模型默认 | 最大序列长度 | 按业务需求设置，过大浪费显存 |
| `--gpu-memory-utilization` | 0.9 | GPU 显存利用率 | 生产环境建议 0.85-0.92 |
| `--block-size` | 16 | 每个物理块的 token 数 | 通常不需要修改 |
| `--swap-space` | 4 | CPU swap 空间 (GB) | 高并发场景可增加 |
| `--max-num-seqs` | 256 | 最大并发序列数 | 根据显存和延迟要求调整 |
| `--max-num-batched-tokens` | 无限制 | 单步最大 token 数 | 控制 prefill 批大小 |

### 5.2 调度参数

| 参数 | 默认值 | 含义 | 调优建议 |
|------|--------|------|----------|
| `--scheduler-delay-factor` | 0.0 | 调度延迟因子 | 增大可提高批处理效率但增加延迟 |
| `--enable-chunked-prefill` | False | 启用分块 prefill | 长序列场景建议开启 |
| `--max-num-batched-tokens` | - | chunked prefill 块大小 | 开启 chunked prefill 时设为 512-2048 |
| `--preemption-mode` | "recompute" | 抢占模式 | swap 模式适合长序列 |

### 5.3 性能参数

| 参数 | 默认值 | 含义 | 调优建议 |
|------|--------|------|----------|
| `--enforce-eager` | False | 禁用 CUDA graph | 调试时开启，生产关闭 |
| `--max-seq-len-to-capture` | 8192 | CUDA graph 捕获的最大序列长度 | 超过此长度回退 eager 模式 |
| `--enable-prefix-caching` | False | 启用前缀缓存 | 多轮对话/共享 system prompt 时开启 |
| `--quantization` | None | 量化方式 | awq/gptq/squeezellm |
| `--dtype` | "auto" | 数据类型 | float16/bfloat16 |

## 6. 调优目标

### 6.1 吞吐量优先场景

目标：最大化单位时间内生成的 token 总数（tokens/s）

策略：
- 增大 `--max-num-seqs` 以提高批处理并发
- 增大 `--gpu-memory-utilization` 到 0.92-0.95
- 启用 prefix caching 减少重复计算
- 使用量化（AWQ/GPTQ）减少模型显存占用，腾出空间给 KV cache
- 关闭 `--enforce-eager`，启用 CUDA graph

权衡：
- 吞吐量提升通常伴随单请求延迟增加
- 更大的 batch size 意味着更高的 TPOT
- 显存利用率过高可能导致 OOM 或频繁 preemption

### 6.2 延迟优先场景

目标：最小化 TTFT 和 TPOT

策略：
- 限制 `--max-num-seqs` 为较小值（16-64）
- 启用 chunked prefill 避免长 prefill 阻塞 decode
- 设置 `--max-num-batched-tokens` 限制单步计算量
- 使用更高的 `--tensor-parallel-size` 分摊计算

权衡：
- 低延迟配置下 GPU 利用率可能不足 50%
- 限制并发数会降低整体吞吐量
- TP 增加引入通信开销

### 6.3 成本优先场景

目标：最大化 tokens/$/hour

策略：
- 使用量化模型（INT4/INT8）减少所需 GPU 数量
- 合理设置 `--max-model-len` 避免过度预留
- 启用 prefix caching 减少重复计算
- 选择性价比最高的 GPU（如 A10G vs A100）

## 7. 适用场景

| 场景 | 适用性 | 原因 |
|------|--------|------|
| 在线 API 服务 | 非常适合 | OpenAI 兼容 API + continuous batching |
| 批量离线推理 | 适合 | 高吞吐量 + 自动批处理 |
| 多轮对话 | 非常适合 | prefix caching 复用历史 KV cache |
| 长文本生成 | 适合 | PagedAttention 高效管理大 KV cache |
| 实时交互（<100ms TPOT） | 需要调优 | 需要限制 batch size + TP |
| 边缘设备部署 | 不适合 | 依赖高端 GPU，无 CPU/移动端支持 |
| 模型微调 | 不适合 | vLLM 专注推理，不支持训练 |

## 8. 不适用场景

1. **单请求低延迟极致优化**：vLLM 的调度开销在极低延迟场景下不可忽略，TensorRT-LLM 更适合
2. **CPU/边缘推理**：vLLM 仅支持 NVIDIA GPU，llama.cpp 更适合
3. **小模型（<1B）**：调度和内存管理开销相对模型计算过大
4. **需要自定义 attention kernel**：vLLM 的 PagedAttention kernel 不易替换
5. **Windows 原生部署**：官方仅支持 Linux

## 9. 副作用

### 9.1 显存碎片化（长期运行）

虽然 PagedAttention 大幅减少碎片，但长期运行后物理块的分配模式可能导致：
- 连续大块分配失败
- swap 频率增加
- 需要定期重启服务

### 9.2 调度延迟抖动

continuous batching 的调度决策引入不确定性：
- 新请求的 prefill 可能延迟正在 decode 的请求
- preemption 导致被抢占请求的延迟突增
- 批次大小波动导致 TPOT 不稳定

### 9.3 CUDA Graph 内存开销

启用 CUDA graph 后：
- 每个 batch size 需要预录制一个 graph，占用额外显存
- 不支持动态 shape，超出预录制范围回退 eager 模式
- 首次请求有 warmup 延迟

## 10. 风险

### 10.1 OOM（Out of Memory）

触发条件：
- `gpu-memory-utilization` 设置过高
- 突发大量长序列请求
- prefix caching 占用过多块

缓解措施：
- 设置合理的 `gpu-memory-utilization`（0.85-0.90）
- 配置 `max-num-seqs` 上限
- 监控显存使用率并设置告警

### 10.2 请求饿死（Starvation）

触发条件：
- 持续高负载下，新请求长时间在 waiting 队列
- preemption 后的请求反复被抢占

缓解措施：
- 设置请求超时
- 实现优先级队列
- 监控 waiting 队列长度

### 10.3 性能退化

触发条件：
- 模型更新后 CUDA graph 失效
- 序列长度分布突变
- 并发模式变化

缓解措施：
- 部署后执行 warmup
- 持续监控 P99 延迟
- A/B 测试新配置

## 11. 验证方式

### 11.1 功能验证

```bash
# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096

# 发送测试请求
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 100}'
```

### 11.2 性能验证

```bash
# 使用 vLLM 内置 benchmark
python -m vllm.entrypoints.openai.bench_serving \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 1000 \
    --request-rate 10

# 关键指标检查
# - Throughput: tokens/s
# - Mean TTFT: ms
# - Mean TPOT: ms  
# - P99 latency: ms
```

### 11.3 稳定性验证

```bash
# 长时间压测（至少 1 小时）
# 监控指标：
# - 显存使用是否持续增长（内存泄漏）
# - P99 延迟是否逐渐恶化
# - 错误率是否上升
# - preemption 频率是否异常
```

## 12. 监控指标

### 12.1 核心业务指标

| 指标 | 采集方式 | 告警阈值建议 |
|------|----------|--------------|
| TTFT P50/P99 | Prometheus metrics | P99 > 2x baseline |
| TPOT P50/P99 | Prometheus metrics | P99 > 3x baseline |
| Throughput (tokens/s) | Prometheus metrics | < 70% baseline |
| 请求成功率 | HTTP status code | < 99.5% |
| 队列等待时间 | Scheduler metrics | > 5s |

### 12.2 系统资源指标

| 指标 | 采集方式 | 告警阈值建议 |
|------|----------|--------------|
| GPU 利用率 | nvidia-smi / DCGM | < 30%（资源浪费） |
| GPU 显存使用率 | nvidia-smi / DCGM | > 95%（OOM 风险） |
| KV cache 使用率 | vLLM metrics | > 90% |
| CPU 内存使用 | node_exporter | > 85% |
| 网络带宽 | node_exporter | 接近上限 |

### 12.3 vLLM 内部指标

```python
# vLLM 暴露的 Prometheus 指标
vllm:num_requests_running      # 正在运行的请求数
vllm:num_requests_waiting      # 等待中的请求数
vllm:num_requests_swapped      # 被换出的请求数
vllm:gpu_cache_usage_perc      # GPU KV cache 使用率
vllm:cpu_cache_usage_perc      # CPU KV cache 使用率
vllm:num_preemptions_total     # 抢占总次数
vllm:prompt_tokens_total       # 输入 token 总数
vllm:generation_tokens_total   # 生成 token 总数
vllm:time_to_first_token_seconds  # TTFT 直方图
vllm:time_per_output_token_seconds # TPOT 直方图
```

## 13. 压测方法

### 13.1 基准压测

```bash
# 固定输入长度 + 固定输出长度
python benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 500 \
    --request-rate inf \
    --input-len 512 \
    --output-len 128
```

### 13.2 真实负载模拟

```bash
# 使用 ShareGPT 数据集模拟真实分布
python benchmark_serving.py \
    --backend vllm \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered.json \
    --num-prompts 1000 \
    --request-rate 5  # 模拟实际 QPS
```

### 13.3 极限压测

```bash
# 逐步增加 request rate 直到系统饱和
for rate in 1 2 5 10 20 50 100; do
    python benchmark_serving.py \
        --request-rate $rate \
        --num-prompts 200 \
        --output-file results_rate_${rate}.json
done
# 绘制 throughput vs latency 曲线，找到拐点
```

### 13.4 长序列压测

```bash
# 测试长上下文场景
python benchmark_serving.py \
    --input-len 8192 \
    --output-len 2048 \
    --num-prompts 50 \
    --request-rate 2
```

## 14. Profiling 方法

### 14.1 PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/vllm'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # 执行推理步骤
    engine.step()
```

### 14.2 NVIDIA Nsight Systems

```bash
# 捕获 GPU 活动
nsys profile -t cuda,nvtx,osrt \
    -o vllm_profile \
    --force-overwrite true \
    python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct

# 分析结果
nsys stats vllm_profile.nsys-rep
```

### 14.3 vLLM 内置 Profiling

```bash
# 启用详细日志
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server ...

# 启用 trace
export VLLM_TRACE_FUNCTION=1
```

## 15. 失败案例

### 案例 1：OOM 导致服务崩溃

**现象**：服务运行 2 小时后突然 OOM 崩溃
**根因**：`gpu-memory-utilization=0.95` 过高，加上 prefix caching 持续积累，无法应对突发长序列
**修复**：降低到 0.88，设置 prefix cache 的 eviction 策略
**教训**：生产环境必须预留 10-15% 显存余量

### 案例 2：TTFT 突增

**现象**：P99 TTFT 从 200ms 突增到 2000ms
**根因**：大量长序列请求同时到达，prefill 阻塞了 decode 批次
**修复**：启用 chunked prefill，设置 `max-num-batched-tokens=2048`
**教训**：必须监控 prefill 队列长度，设置 chunked prefill 作为保护

### 案例 3：吞吐量远低于预期

**现象**：8B 模型在 A100 上仅 500 tokens/s，预期 2000+
**根因**：`enforce-eager=True` 未关闭，CUDA graph 未启用；`max-num-seqs=8` 过小
**修复**：关闭 enforce-eager，增大 max-num-seqs 到 128
**教训**：部署前必须执行 benchmark 对比默认配置

## 16. 复盘模板

```markdown
## 事件复盘：[事件标题]

### 时间线
- [时间] 发现异常
- [时间] 开始排查
- [时间] 定位根因
- [时间] 实施修复
- [时间] 验证恢复

### 影响范围
- 受影响用户数：
- 受影响请求数：
- 持续时间：

### 根因分析
- 直接原因：
- 根本原因：
- 触发条件：

### 修复措施
- 临时措施：
- 永久修复：

### 预防措施
- 监控改进：
- 配置变更：
- 流程改进：

### 经验教训
- 做得好的：
- 需要改进的：
- Action Items：
```

## 17. 实验任务

### 实验 1：基础部署与 Benchmark

1. 使用 vLLM 部署 Llama-3.1-8B-Instruct
2. 执行基准 benchmark，记录 baseline 指标
3. 调整 `max-num-seqs` 从 8 到 256，观察吞吐量和延迟变化
4. 绘制 throughput vs latency 曲线

### 实验 2：显存管理实验

1. 对比 `gpu-memory-utilization` 为 0.7/0.8/0.9/0.95 时的表现
2. 观察不同设置下的 preemption 频率
3. 测试 OOM 边界条件
4. 记录 KV cache 使用率随时间的变化

### 实验 3：Prefix Caching 效果验证

1. 准备共享 system prompt 的多轮对话数据集
2. 对比开启/关闭 prefix caching 的 TTFT
3. 测量 prefix cache hit rate
4. 分析不同 prefix 长度下的收益

### 实验 4：Chunked Prefill 调优

1. 对比开启/关闭 chunked prefill 的 TTFT 和 TPOT
2. 测试不同 chunk size（512/1024/2048/4096）的效果
3. 在混合长短请求场景下验证效果
4. 找到最优 chunk size

### 实验 5：Tensor Parallel 扩展性

1. 对比 TP=1/2/4 的吞吐量和延迟
2. 测量 TP 通信开销占比
3. 分析不同模型大小下 TP 的收益曲线
4. 确定最优 TP 配置

## 18. 习题 20 道

### 基础概念（1-5）

**Q1**：vLLM 中 PagedAttention 的核心思想是什么？它解决了传统推理框架的什么问题？

**Q2**：解释 vLLM Scheduler 的三个队列（Waiting、Running、Swapped）的作用和优先级关系。

**Q3**：什么是 preemption？vLLM 支持哪两种 preemption 模式？各自的优缺点是什么？

**Q4**：vLLM 的 Block 大小（block_size）如何影响显存利用率和性能？为什么默认值是 16？

**Q5**：解释 vLLM 中 logical block 和 physical block 的区别，以及 Copy-on-Write 机制的作用。

### 性能调优（6-10）

**Q6**：一个在线服务要求 P99 TTFT < 500ms，当前 P99 TTFT 为 1200ms。列出至少 3 种可能的优化方向。

**Q7**：`gpu-memory-utilization` 设置为 0.95 和 0.85 分别适合什么场景？各自的风险是什么？

**Q8**：在什么情况下应该启用 chunked prefill？它对 TTFT 和 throughput 分别有什么影响？

**Q9**：如何判断 vLLM 服务是 compute-bound 还是 memory-bound？对应的优化策略有何不同？

**Q10**：CUDA graph 在 vLLM 中的作用是什么？什么情况下应该禁用它？

### 生产运维（11-15）

**Q11**：vLLM 服务运行 24 小时后 P99 延迟逐渐增加，可能的原因有哪些？如何排查？

**Q12**：如何设计 vLLM 服务的健康检查（health check）？需要检查哪些维度？

**Q13**：在多实例部署中，如何实现请求的负载均衡？简单轮询有什么问题？

**Q14**：vLLM 服务突然出现大量 503 错误，列出排查步骤。

**Q15**：如何实现 vLLM 服务的无损滚动更新（rolling update）？

### 架构设计（16-20）

**Q16**：对比 vLLM 和 TensorRT-LLM 的架构差异，各自适合什么场景？

**Q17**：如果需要支持 100+ QPS 的在线服务，如何设计 vLLM 的部署架构？

**Q18**：vLLM 的 AsyncLLMEngine 和 LLMEngine 的区别是什么？为什么在线服务必须使用 AsyncLLMEngine？

**Q19**：如何在 vLLM 中实现请求优先级？当前版本是否原生支持？

**Q20**：设计一个支持多模型、多版本的 vLLM 服务架构，需要考虑哪些因素？

## 19. 标准答案

### A1
PagedAttention 的核心思想是将 KV cache 按固定大小的块（block）进行管理，类似操作系统的虚拟内存分页机制。它解决了传统框架中 KV cache 必须连续分配、按最大长度预留导致的显存碎片化和浪费问题。通过分页管理，KV cache 可以非连续存储，按需分配，显存利用率从约 50% 提升到 95% 以上。

### A2
- **Waiting 队列**：存放新到达的请求，等待 prefill 执行。优先级最低。
- **Running 队列**：存放正在 decode 的请求。优先级最高，因为已经投入了 prefill 计算资源。
- **Swapped 队列**：存放被抢占换出到 CPU 的请求。优先级中等，恢复比重新 prefill 成本低。

调度顺序：Running > Swapped > Waiting。这确保已投入资源的请求优先完成。

### A3
Preemption 是当显存不足时，暂停某些正在运行的请求以腾出资源的机制。

- **Recompute 模式**：丢弃被抢占请求的 KV cache，恢复时重新 prefill。优点是不需要 CPU 内存；缺点是恢复成本高。
- **Swap 模式**：将 KV cache 换出到 CPU 内存，恢复时换回。优点是恢复快；缺点是需要 CPU 内存，且 PCIe 带宽可能成为瓶颈。

### A4
Block size 决定了 KV cache 的最小分配粒度：
- 过小（如 1）：元数据开销大，block table 过长
- 过大（如 64）：最后一个 block 的内部碎片大
- 默认 16 是在碎片率和管理开销之间的平衡点
- 对于短序列场景可以考虑减小，长序列场景可以增大

### A5
- **Logical block**：每个序列拥有的逻辑地址空间中的块，从 0 开始编号
- **Physical block**：GPU 显存中实际分配的块，全局编号
- 映射关系通过 block table 维护
- **Copy-on-Write**：beam search 等场景下，多个序列共享相同的 physical block。当某个序列需要修改共享块时，才复制一份新的 physical block。减少了显存使用。

### A6
优化 TTFT 的方向：
1. 启用 chunked prefill，将长 prefill 分块执行，避免阻塞
2. 增加 tensor parallel 度，加速 prefill 计算
3. 限制 `max-num-batched-tokens`，控制单步 prefill 量
4. 启用 prefix caching，复用已计算的 KV cache
5. 检查是否有大量请求排队，增加实例数或优化调度

### A7
- **0.95**：适合离线批处理、对延迟不敏感的场景。风险是突发长序列可能 OOM，preemption 频繁。
- **0.85**：适合在线服务、延迟敏感场景。预留 15% 余量应对突发，但牺牲了部分吞吐量。
- 生产建议：在线服务 0.85-0.90，离线批处理 0.90-0.95。

### A8
启用 chunked prefill 的场景：
- 输入序列长度差异大（有长有短）
- 对 TTFT 有严格 SLA 要求
- decode 请求不能被长 prefill 阻塞

影响：
- TTFT：对短请求改善不大，对长请求可能略增（分多步完成）
- Throughput：略有下降（调度开销增加），但整体延迟分布更均匀
- 核心收益是避免长 prefill 导致的 decode 延迟尖刺

### A9
判断方法：
- **Compute-bound**：GPU 利用率高（>80%），增加 batch size 不提升吞吐。常见于 prefill 阶段。
- **Memory-bound**：GPU 利用率低（<50%），显存带宽接近上限。常见于 decode 阶段。

优化策略：
- Compute-bound：使用量化减少计算量，增加 TP 分摊计算
- Memory-bound：增大 batch size 提高计算/访存比，使用 CUDA graph 减少 launch 开销

### A10
CUDA graph 的作用：将多个 CUDA kernel 的 launch 序列预录制为一个 graph，执行时一次性提交，减少 CPU-GPU 交互开销。在 decode 阶段（kernel 小而多）效果显著，可提升 10-30% 吞吐。

禁用场景：
- 调试阶段（需要逐 kernel 检查）
- 显存极度紧张（graph 占用额外显存）
- 序列长度高度动态（超出预录制范围频繁回退）
- 使用自定义 kernel 不兼容 graph capture

### A11
可能原因：
1. KV cache 碎片化累积
2. Prefix cache 占用过多块，有效可用块减少
3. 内存泄漏（Python 对象未释放）
4. 请求模式变化（序列变长）
5. GPU 温度过高导致降频

排查步骤：
1. 检查 `gpu_cache_usage_perc` 趋势
2. 检查 `num_preemptions_total` 增长率
3. 检查 GPU 温度和频率
4. 对比请求长度分布变化
5. 重启服务后是否恢复

### A12
健康检查维度：
1. **存活检查（Liveness）**：进程是否存在，API 是否响应
2. **就绪检查（Readiness）**：模型是否加载完成，能否处理请求
3. **深度检查**：KV cache 使用率 < 95%，waiting 队列 < 阈值，最近 1 分钟无 OOM
4. **性能检查**：P99 延迟在 SLA 范围内

### A13
简单轮询的问题：不考虑各实例的当前负载，可能将请求发到已经过载的实例。

更好的策略：
- **最少连接（Least Connections）**：发到当前处理请求最少的实例
- **加权负载感知**：根据 KV cache 使用率、队列长度加权
- **前缀亲和性**：相同 prefix 的请求路由到同一实例，提高 cache hit rate

### A14
排查步骤：
1. 检查 GPU 显存是否 OOM → 查看 dmesg 和 vLLM 日志
2. 检查 waiting 队列是否爆满 → 查看 `num_requests_waiting`
3. 检查是否有大量 preemption → 查看 `num_preemptions_total`
4. 检查网络/磁盘是否异常 → 查看系统指标
5. 检查请求是否异常（超长输入）→ 查看请求日志
6. 检查 GPU 是否故障 → nvidia-smi 检查 ECC 错误

### A15
无损滚动更新步骤：
1. 启动新实例，等待模型加载完成（readiness check 通过）
2. 将新实例加入负载均衡池
3. 将旧实例标记为 draining（不接受新请求）
4. 等待旧实例上所有正在处理的请求完成（设置超时）
5. 从负载均衡池移除旧实例
6. 关闭旧实例

关键：需要足够的超时时间让长请求完成，同时设置硬超时避免无限等待。

### A16
| 维度 | vLLM | TensorRT-LLM |
|------|------|--------------|
| 开发语言 | Python + CUDA | C++ + CUDA |
| 易用性 | 高，pip install 即用 | 低，需要编译模型 |
| 性能 | 优秀 | 极致（通常快 10-30%） |
| 灵活性 | 高，支持多种模型 | 中，需要适配 |
| 适合场景 | 快速迭代、多模型 | 极致性能、固定模型 |
| 社区 | 活跃开源 | NVIDIA 主导 |

### A17
100+ QPS 部署架构：
1. 多实例部署（根据单实例 QPS 上限计算实例数）
2. 前置负载均衡器（Nginx/Envoy + 加权路由）
3. 请求队列（处理突发流量）
4. 自动扩缩容（基于 QPS 和延迟指标）
5. 前缀亲和性路由（提高 cache hit rate）
6. 监控告警体系（Prometheus + Grafana）
7. 灰度发布能力

### A18
- **LLMEngine**：同步引擎，调用 `generate()` 会阻塞直到完成。适合离线批处理。
- **AsyncLLMEngine**：异步引擎，基于 asyncio，支持并发处理多个请求。在线服务必须使用。

原因：在线服务需要同时处理多个请求，同步引擎会导致请求串行化，无法利用 continuous batching 的优势。

### A19
当前 vLLM（截至 2025 年底）通过 `priority` 参数支持基本的请求优先级。实现方式：
- Scheduler 在选择 waiting 队列中的请求时，按 priority 排序
- 高优先级请求可以触发低优先级请求的 preemption
- 自定义实现：可以在 API 层实现优先级队列，控制请求提交顺序

### A20
多模型多版本架构考虑因素：
1. **模型隔离**：每个模型独立实例 vs 共享实例（LoRA adapter）
2. **版本管理**：模型版本注册表，支持回滚
3. **路由策略**：按模型名 + 版本路由到对应实例
4. **资源分配**：不同模型的 GPU 资源配额
5. **缓存策略**：模型间是否共享 prefix cache
6. **监控粒度**：按模型 + 版本维度的指标
7. **发布流程**：金丝雀发布、A/B 测试

## 20. 调优 Checklist

### 部署前

- [ ] 确认模型大小与 GPU 显存匹配
- [ ] 确定 tensor parallel 度（模型放不下单卡时）
- [ ] 设置合理的 `max-model-len`（不超过业务需求）
- [ ] 选择合适的量化方式（如需要）
- [ ] 设置 `gpu-memory-utilization`（在线 0.85-0.90，离线 0.90-0.95）
- [ ] 配置 `max-num-seqs`（根据延迟要求）
- [ ] 决定是否启用 prefix caching
- [ ] 决定是否启用 chunked prefill

### 部署后验证

- [ ] 执行 warmup 请求（触发 CUDA graph 录制）
- [ ] 运行 benchmark 确认性能达标
- [ ] 验证 TTFT 和 TPOT 满足 SLA
- [ ] 确认显存使用稳定
- [ ] 检查错误率为 0

### 生产运行

- [ ] 配置 Prometheus 指标采集
- [ ] 设置关键指标告警（TTFT P99、错误率、显存）
- [ ] 配置健康检查（liveness + readiness）
- [ ] 设置请求超时
- [ ] 配置日志收集
- [ ] 制定扩缩容策略
- [ ] 制定回滚方案

### 定期维护

- [ ] 每周检查 P99 延迟趋势
- [ ] 每月评估是否需要扩容
- [ ] 版本更新前在 staging 环境验证
- [ ] 定期清理 prefix cache（如果持续增长）
- [ ] 检查 GPU 健康状态（ECC 错误、温度）
