# 综合 Mock 评分与补课清单

## 学习目标

本章用于把一次 90 分钟综合 mock 变成可复盘、可量化、可补课的训练流程。完成后应能:

1. 按系统设计、CUDA、框架、基础四类能力评分。
2. 用 1-5 分制描述表现, 避免只写"感觉还行"。
3. 执行固定流程: 45min 系统设计 + 30min CUDA coding + 15min 基础问答。
4. 把问题转成一周内可验收的补课清单。
5. 用 checklist 找出下一次训练最该提升的 3 个点。

## 前置知识

1. LLM serving: rate limit、KV cache、batch scheduler、streaming、observability。
2. CUDA: vector add、reduction、softmax、GEMM tiling。
3. Framework: PyTorch extension、Triton launch、operator test、benchmark。
4. 算法: heap、beam search、prefix tree、queue、token bucket。
5. 表达: 先澄清约束, 再给 baseline, 最后讲 tradeoff 和风险。

## 核心内容

综合 mock 模拟真实面试的约束:

1. 时间有限, 必须先完成主路径。
2. 信息不完整, 需要主动问 QPS、tokens、SLO、显存等约束。
3. 追问会改变方向, baseline 要稳, 扩展要有边界。
4. 评分看行为证据: 说过什么、写出什么、测了什么、承认了什么。

四个评分维度:

1. 系统设计: 从需求到架构, 再到资源和故障。
2. CUDA: 写正确 kernel, 解释 memory/sync/perf。
3. 框架: 理解 PyTorch/Triton/operator 集成和验证。
4. 基础: 算法、数据结构、OS/并发和数值稳定。

## 完整的问答/题目

### Mock 总流程

总时长: 90 分钟。

1. 0-5 分钟: 面试官给场景, 候选人澄清约束。
2. 5-45 分钟: 系统设计主问题。
3. 45-75 分钟: CUDA coding 主问题。
4. 75-90 分钟: 基础问答和追问。
5. 90-105 分钟: 评分复盘, 不计入正式时间。

### Part A: 45min 系统设计

推荐题目: 设计一个 LLM inference service, 支持 chat completion、多租户 rate limit、continuous batching、KV cache 复用和 streaming output。目标是同时控制 TTFT、ITL、吞吐和显存。

候选人应覆盖:

1. API: `POST /v1/chat/completions`, request 包含 messages、max_new_tokens、temperature、stream、priority。
2. Admission: auth 后按 org/user 做 request 和 token limit; prompt token 计数; output token reservation; KV capacity check。
3. Scheduler: waiting queue、running decode set、token budget; decode 优先; prefill chunking。
4. KV cache: page/block allocator; per-sequence block table; prefix cache 需要模型和 tokenizer 版本。
5. Worker: prefill 填 KV; decode 读 KV 生成 token; sampling 后 stream flush。
6. Metrics: QPS、input/output tokens/s、TTFT、ITL、queue delay、GPU utilization、KV used pages、OOM、429。
7. Failure: worker crash、Redis 延迟、GPU OOM、client cancel、partial generation cleanup。

高质量回答骨架:

```text
Client -> API Gateway -> Auth/RateLimit -> Admission Queue
       -> Scheduler -> GPU Worker -> Streamer -> Client
                         |
                         +-> KV Page Allocator
                         +-> Metrics/Tracing
```

面试官追问:

1. 32k prompt 如何避免阻塞所有 decode?
2. streaming client 断开后哪些资源必须释放?
3. Redis 延迟抖动时 rate limiter 如何降级?
4. KV cache 剩余 5%, 新请求如何 admission?
5. 如何定义并监控 TTFT 和 ITL?

### Part B: 30min CUDA Coding

推荐题目: 实现 row-wise softmax 或 block reduction。必须处理非对齐长度。

候选人应覆盖: shape/dtype、thread/block 映射、tail mask、shared memory 或 warp reduction、数值稳定、reference check。

row-wise softmax 最小答案:

```cpp
__global__ void row_softmax(const float* x, float* y, int rows, int cols) {
    extern __shared__ float buf[];
    int row = blockIdx.x, tid = threadIdx.x;
    float v = -INFINITY;
    if (tid < cols) v = x[row * cols + tid];
    buf[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = fmaxf(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    float m = buf[0];
    float e = 0.0f;
    if (tid < cols) e = expf(x[row * cols + tid] - m);
    buf[tid] = e;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid < cols) y[row * cols + tid] = e / buf[0];
}
```

block reduction 最小答案:

```cpp
__global__ void reduce_sum(const float* x, float* partial, int n) {
    extern __shared__ float buf[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    float v = 0.0f;
    if (idx < n) v += x[idx];
    if (idx + blockDim.x < n) v += x[idx + blockDim.x];
    buf[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = buf[0];
}
```

测试要求: aligned 如 `cols=1024` 或 `n=4096`; non-aligned 如 `cols=777` 或 `n=1003`; 随机值含负数和较大正数; PyTorch reference; float32 tolerance 如 `rtol=1e-4, atol=1e-5`。

### Part C: 15min 基础问答

1. 为什么 softmax 要减 max? A: 防 overflow, softmax 平移不变, NaN/Inf 要定义策略。
2. top-k sampling 和 beam search 目标有什么不同? A: sampling 追求随机多样性, beam search 搜索高概率序列。
3. KV cache 大小和哪些参数相关? A: layers、heads、head_dim、dtype、sequence length、batch size、K/V 两份。
4. continuous batching 为什么不能只按 request count? A: token 数更接近 GPU 计算和显存成本。
5. shared memory race 怎么出现? A: 多线程写后被其他线程读, 缺 barrier 或 barrier 分支不一致。
6. PyTorch extension 如何验证? A: reference、aligned/non-aligned、dtype/device、grad 如需要、benchmark。
7. 什么是 coalesced access? A: 同一 warp 相邻线程访问连续地址, 减少 memory transaction。
8. output tokens 为什么要预留? A: 生成前未知长度, 预留防超卖, 结束后退还。

评分记录模板:

```text
候选人:
日期:
题目:
系统设计分数/证据:
CUDA 分数/证据:
框架分数/证据:
基础分数/证据:
总评:
下一次最该练的 3 件事:
1.
2.
3.
```

## 追问方向与深入点

系统设计:

1. 多租户如何避免异常客户占满 batch?
2. cancel 如何从 API 传播到 scheduler、worker、KV allocator?
3. Prefix cache 如何验证模型版本、tokenizer 版本和 adapter?
4. Chunked prefill 的 chunk size 如何动态调整?
5. TTFT 和 ITL 冲突时如何做优先级策略?
6. 队列满时返回 429、503 还是降级?
7. 哪些指标能定位 tokenizer 慢、排队慢、prefill 慢或 decode 慢?

CUDA:

1. reduction 如何用 warp shuffle 优化最后 32 个元素?
2. softmax 对 `cols > 1024` 怎么办?
3. GEMM baseline 到 tensor core 需要补哪些概念?
4. blockDim 不是 2 的幂时 reduction 怎么写?
5. conditional barrier 为什么可能死锁?
6. benchmark 如何避免只测 launch overhead?
7. Triton mask 和 CUDA if guard 如何对应?

框架与基础:

1. PyTorch custom op 的 C++ binding、CUDA kernel、Python wrapper 如何分层?
2. Triton block size 如何 autotune?
3. benchmark 为什么要 warmup 和 synchronize?
4. extension 失败时如何 fallback 到 PyTorch reference?
5. heap top-k 和 quickselect top-k 的 tradeoff 是什么?
6. prefix tree 删除节点如何处理共享前缀?
7. token bucket 和 leaky bucket 的区别是什么?

## 评分标准

### 维度权重

总分 100: 系统设计 35, CUDA 30, 框架 20, 基础 15。Kernel 岗可调整为 CUDA 40、系统设计 25、框架 20、基础 15。

### 系统设计 1-5 分制

1 分: 只描述单机请求流程, 没有 rate limit、scheduler、KV cache、失败处理。

2 分: 能画 API -> worker -> GPU, 但 batch、KV、限流和资源控制粗糙。

3 分: 覆盖 admission、continuous batching、KV page allocator、streaming 和核心指标。

4 分: 能解释 token budget、chunked prefill、decode priority、分布式限流、资源释放和降级。

5 分: 数据结构、状态机、并发边界、metrics、SLO、容量估算和故障恢复都清楚。

### CUDA 1-5 分制

1 分: 线程映射错误或明显越界, block/thread/grid 解释不清。

2 分: baseline 接近正确, 但 tail mask、同步或数值稳定有关键漏洞。

3 分: 正确处理普通和非对齐输入, 能用 PyTorch reference 验证。

4 分: 能讨论 coalescing、occupancy、warp primitive、bank conflict、tolerance 和 benchmark。

5 分: 能从 baseline 推进到生产级优化, 包括 Triton/CUDA 差异、profile 指标和 shape 策略。

### 框架 1-5 分制

1 分: 只会写孤立 kernel, 不知道如何接入 PyTorch/Triton。

2 分: 知道 wrapper 和编译扩展, 但缺 dtype/device/shape check、fallback、benchmark。

3 分: 能描述 C++ binding、CUDA launch、Python API、pytest reference check。

4 分: 能处理动态 shape、contiguous check、error message、无 GPU CI 跳过和 benchmark 同步。

5 分: 能放进 inference framework: dispatch、autotune、fallback、profiling、版本兼容和观测完整。

### 基础 1-5 分制

1 分: 算法和数据结构概念混乱, 复杂度说不清。

2 分: 能写简单实现, 但边界、复杂度或数值稳定常遗漏。

3 分: top-k、beam search、prefix tree、token bucket 能正确实现并说明复杂度。

4 分: 能把算法连接到 prompt cache、sampling pipeline、scheduler queue。

5 分: 能比较多种方案, 识别工程限制, 追问下推理清晰。

综合评级: 4.5-5.0 强通过; 3.5-4.4 通过倾向; 2.5-3.4 边缘; 1.5-2.4 不通过; 1.0-1.4 明显不匹配。

## 补课清单模板

```text
Mock 日期:
总分:
最大短板:

问题 1:
证据:
补课动作:
验收标准:
截止日期:

问题 2:
证据:
补课动作:
验收标准:
截止日期:

问题 3:
证据:
补课动作:
验收标准:
截止日期:
```

补课动作示例:

1. TTFT/ITL 不清楚: 重画 prefill/decode timeline; 验收为 3 分钟内解释 queue delay、prefill、first decode、stream flush。
2. KV allocator 只会连续分配: 手写 page allocator 和 block table; 验收为能处理 allocate、append、free、OOM、shared prefix。
3. tail mask 经常遗漏: 每个 kernel 写 aligned/non-aligned 测试; 验收为指出每处 global load/store 的 mask。
4. barrier 不稳定: 对 reduction/softmax 标出数据依赖; 验收为解释删除任一 barrier 的 race。
5. extension 验证薄弱: 写 shape、dtype、device、reference、tolerance、benchmark checklist。
6. top-k 和 beam 混淆: 各写 30 行 Python; 验收为能解释随机性、长度惩罚和 eos。

## 自评 checklist

系统设计:

1. 是否先问 QPS、prompt length、output length、SLO、GPU memory?
2. 是否明确区分 prefill 和 decode?
3. 是否使用 token budget 而不是只说 batch size?
4. 是否说明 KV cache 的数据结构和释放时机?
5. 是否处理 streaming client cancel?
6. 是否给出 TTFT、ITL、tokens/s、OOM、429 等 metrics?
7. 是否说明 rate limit 的分布式原子性?
8. 是否解释 Redis/GPU/worker 故障下如何降级?

CUDA:

1. 是否写清 grid、block、thread 到数据的映射?
2. 每个 global memory load/store 是否有边界保护?
3. 是否解释 shared memory 生命周期?
4. 每个 barrier 是否有明确数据依赖?
5. softmax 是否做 max subtraction?
6. reduction 是否说明 partial sum 第二阶段?
7. GEMM 是否 mask A load、B load、C store?
8. 是否给出 aligned 和 non-aligned 测试?

框架与基础:

1. 是否说明 Python API、C++ binding、CUDA launch 边界?
2. 是否检查 dtype、device、contiguous、shape?
3. 是否有 PyTorch reference fallback?
4. benchmark 是否 warmup 和 synchronize?
5. 是否能写 top-k sampling、beam search、longest-prefix-match?
6. 是否能说明 token bucket refill 和 atomic take?
7. 是否能给出时间复杂度和空间复杂度?
8. 是否处理空输入、非法参数、eos、重复 token 等边界?

## 复习卡片 15 张

1. Q: 为什么固定 45+30+15? A: 模拟系统、coding、基础追问的真实时间压力。
2. Q: 系统设计 3 分和 4 分差别是什么? A: 3 分主路径完整, 4 分能处理资源冲突、降级、指标和 tradeoff。
3. Q: CUDA 2 分常见问题是什么? A: baseline 接近正确, 但 tail mask、barrier 或数值稳定有漏洞。
4. Q: 为什么单独评分框架? A: 面试看能否把 kernel 接入 PyTorch/Triton 并验证、benchmark、fallback。
5. Q: 基础如何连接 LLM serving? A: top-k 对应 sampling, beam 对应 decoding, prefix tree 对应 prompt cache。
6. Q: TTFT 包含哪些阶段? A: 排队、admission、prefill、第一次 decode、stream flush。
7. Q: ITL 受什么影响? A: decode scheduling、batch composition、GPU step time、stream flush 和抖动。
8. Q: token budget 为什么优于 request count? A: token 数更接近计算和显存成本。
9. Q: CUDA reference check 覆盖什么? A: aligned/non-aligned shape、随机值、边界值、dtype/device、tolerance。
10. Q: benchmark 为什么 synchronize? A: CUDA 异步执行, 不同步会只测 launch 或错误时间。
11. Q: rate limit store 故障的取舍是什么? A: fail-open 保护可用性但可能超卖, fail-closed 保护资源但伤可用性。
12. Q: KV OOM 为什么要 cleanup? A: 已接收请求可能占用队列和 cache, 必须释放并 backpressure。
13. Q: conditional barrier 风险是什么? A: block 内不是所有线程到达 barrier 时 kernel 可能挂住。
14. Q: 补课清单为什么写验收标准? A: 没有验收标准的问题无法衡量, 下次 mock 仍会重复。
15. Q: checklist 什么时候填? A: mock 后立即填一次, 隔天复盘再填一次, 区分表达问题和知识问题。
