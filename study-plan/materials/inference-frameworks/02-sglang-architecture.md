# SGLang 架构

## 1. 学习目标

- 理解 SGLang 的核心创新：RadixAttention 与前缀缓存
- 掌握 SGLang 与 vLLM 的架构差异与适用场景
- 理解 constrained decoding（结构化输出）的实现机制
- 掌握 Mooncake 外部 KV Cache 与 RBG（Remote Backend Generation）分离架构
- 能够根据业务场景选择 SGLang vs vLLM

## 2. 系统动机

### 2.1 vLLM 的局限

- Prefix caching 需要手动管理，不够自动化
- 多轮对话和 few-shot 场景下前缀复用效率低
- 缺乏原生的结构化输出支持
- PD（Prefill-Decode）分离不够灵活

### 2.2 SGLang 的设计目标

- **自动前缀缓存**：通过 RadixAttention 自动识别和复用公共前缀
- **高效结构化输出**：原生支持 JSON schema、regex 约束
- **灵活的分离架构**：支持 PD 分离、外部 KV Cache
- **编程友好**：提供 Python DSL 描述复杂生成逻辑

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| RadixAttention | Radix Attention | 基于 radix tree 的自动前缀缓存机制 |
| Radix Tree | Radix Tree | 压缩前缀树，用于管理 KV cache 的共享前缀 |
| Prefix Caching | Prefix Caching | 缓存公共前缀的 KV cache 避免重复计算 |
| Constrained Decoding | Constrained Decoding | 限制输出符合特定格式（JSON、regex） |
| FSM | Finite State Machine | 有限状态机，用于 constrained decoding |
| Mooncake | Mooncake | 月之暗面的外部 KV Cache 存储系统 |
| RBG | Remote Backend Generation | 远程后端生成，PD 分离架构 |
| PD Disaggregation | Prefill-Decode Disaggregation | 将 prefill 和 decode 分离到不同 GPU |
| Chunked Prefill | Chunked Prefill | 将长 prefill 分块执行，与 decode 交错 |

## 4. 执行流程

### 4.1 RadixAttention 工作流程

```
请求到达 → 计算 token 序列的 hash
         → 在 Radix Tree 中查找最长匹配前缀
         → 命中：复用已缓存的 KV cache，只计算新 token 的 prefill
         → 未命中：完整 prefill，将 KV cache 插入 Radix Tree

Radix Tree 结构示例：
Root
├── "System prompt: You are..." (hash: abc123)
│   ├── "User: Hello" (hash: def456)
│   │   └── "Assistant: Hi!" (hash: ghi789)
│   └── "User: What is..." (hash: jkl012)
└── "System prompt: You are a coder..." (hash: mno345)
```

### 4.2 请求处理流程

```
1. Frontend 接收请求
2. Tokenize → 计算 prefix hash
3. Radix Tree lookup → 确定可复用的 KV cache 长度
4. Scheduler 决定：
   - 如果前缀完全命中 → 只需 decode（或短 prefill）
   - 如果部分命中 → 从命中点开始 prefill
   - 如果未命中 → 完整 prefill
5. 执行 prefill/decode
6. 将新生成的 KV cache 插入 Radix Tree
7. LRU 淘汰策略管理 Radix Tree 大小
```

### 4.3 Constrained Decoding 流程

```
1. 用户指定输出格式（JSON schema / regex）
2. 编译为 FSM（有限状态机）
3. 每步 decode 时：
   a. 模型生成 logits
   b. FSM 根据当前状态确定合法 token 集合
   c. Mask 非法 token（设为 -inf）
   d. 从合法 token 中采样
   e. 更新 FSM 状态
4. 优化：预计算 FSM 转移表，batch 内共享 FSM
```

## 5. 参数解释

| 参数 | 默认值 | 含义 | 调优建议 |
|------|--------|------|---------|
| `--mem-fraction-static` | 0.88 | KV cache 占显存比例 | 模型大时降低 |
| `--chunked-prefill-size` | 8192 | chunked prefill 的 chunk 大小 | 长序列时增大 |
| `--schedule-policy` | "lpm" | 调度策略（lpm=longest prefix match） | 前缀复用场景用 lpm |
| `--enable-radix-cache` | True | 启用 RadixAttention | 多轮对话必开 |
| `--disable-radix-cache` | False | 禁用前缀缓存 | 单次请求场景 |
| `--max-running-requests` | auto | 最大并发请求数 | 根据显存调整 |

## 6. 调优目标

### 6.1 前缀缓存命中率

```
目标：prefix_cache_hit_rate > 80%（多轮对话场景）

监控：
- cache_hit_tokens / total_prefill_tokens
- radix_tree_size / max_cache_size
- eviction_rate
```

### 6.2 TTFT 优化

```
With prefix caching:
TTFT = network_latency + (total_tokens - cached_tokens) × prefill_time_per_token

Without:
TTFT = network_latency + total_tokens × prefill_time_per_token

加速比 = total_tokens / (total_tokens - cached_tokens)
```

### 6.3 结构化输出效率

```
目标：constrained decoding overhead < 5% vs unconstrained

关键指标：
- FSM compilation time（应 < 100ms）
- Per-token mask overhead（应 < 0.1ms）
- Token acceptance rate（FSM 不应过度限制）
```

## 7. 适用场景

1. **多轮对话**：system prompt + 历史对话作为公共前缀
2. **Few-shot learning**：多个请求共享相同的 few-shot examples
3. **批量处理**：相同 prompt template + 不同输入
4. **结构化输出**：JSON API、代码生成、表格填充
5. **Agent 场景**：tool calling 的格式约束

## 8. 不适用场景

1. **完全随机请求**：无公共前缀，RadixAttention 无收益
2. **极短请求**：前缀缓存的管理开销 > 收益
3. **单次请求**：无复用机会
4. **显存极度紧张**：Radix Tree 本身占用额外显存

## 9. 副作用

- Radix Tree 占用额外显存（元数据 + 引用计数）
- LRU 淘汰可能导致缓存抖动（频繁 evict + re-prefill）
- Constrained decoding 增加每 token 延迟（FSM 查询）
- 前缀匹配的 hash 计算有 CPU 开销

## 10. 风险

- Hash 冲突（极低概率）导致错误的 KV cache 复用
- Radix Tree 内存泄漏（引用计数错误）
- FSM 编译失败（复杂 regex/schema）
- 长前缀场景下 eviction 策略不当导致 thrashing

## 11. 验证方式

```python
# 验证前缀缓存效果
import requests

# 第一次请求（cold）
resp1 = requests.post(url, json={"prompt": system_prompt + user_msg_1})
ttft_cold = resp1.headers["x-ttft-ms"]

# 第二次请求（warm，共享 system_prompt）
resp2 = requests.post(url, json={"prompt": system_prompt + user_msg_2})
ttft_warm = resp2.headers["x-ttft-ms"]

assert float(ttft_warm) < float(ttft_cold) * 0.5  # 至少 2x 加速
```

## 12. 监控指标

| 指标 | 来源 | 告警阈值 |
|------|------|---------|
| prefix_cache_hit_rate | SGLang metrics | < 50% (多轮场景) |
| radix_tree_utilization | SGLang metrics | > 95% (需扩容) |
| eviction_rate | SGLang metrics | > 10 evictions/s |
| constrained_decode_overhead_ms | SGLang metrics | > 1ms/token |
| ttft_p99_ms | SGLang metrics | > SLA |
| tpot_p99_ms | SGLang metrics | > SLA |

## 13. 压测方法

```python
# 模拟多轮对话场景
import asyncio
import aiohttp

async def multi_turn_benchmark(base_url, num_sessions=100, turns_per_session=5):
    system_prompt = "You are a helpful assistant..."
    
    async with aiohttp.ClientSession() as session:
        for turn in range(turns_per_session):
            tasks = []
            for s in range(num_sessions):
                # 每个 session 共享 system_prompt + 历史
                prompt = system_prompt + history[s] + new_message[s][turn]
                tasks.append(send_request(session, base_url, prompt))
            
            results = await asyncio.gather(*tasks)
            # 记录 TTFT, TPOT, cache hit rate
```

## 14. Profiling 方法

```bash
# SGLang 内置 profiling
python -m sglang.launch_server --model meta-llama/Llama-3-8B \
    --enable-metrics --metrics-port 9090

# Prometheus 查询
curl http://localhost:9090/metrics | grep -E "cache_hit|ttft|tpot"

# Nsight Systems 分析 kernel 级别
nsys profile --trace=cuda,nvtx python -m sglang.bench_serving ...
```

## 15. 失败案例

### Case 1: 缓存抖动
- 现象：TTFT 不稳定，时快时慢
- 原因：Radix Tree 容量不足，频繁 evict 热门前缀
- 修复：增大 `--mem-fraction-static` 或减少并发

### Case 2: Constrained decoding 超时
- 现象：结构化输出请求超时
- 原因：复杂 regex 编译的 FSM 状态数爆炸
- 修复：简化 schema，或预编译 FSM

### Case 3: 前缀不匹配
- 现象：cache hit rate 为 0
- 原因：每次请求的 tokenization 结果不同（如 chat template 变化）
- 修复：确保相同前缀的 tokenization 一致

## 16. 复盘模板

```markdown
## 问题描述
- 现象：
- 影响范围：
- 持续时间：

## 根因分析
- 直接原因：
- 根本原因：
- 为什么没有提前发现：

## 解决方案
- 临时措施：
- 长期方案：
- 验证方式：

## 改进项
- 监控补充：
- 告警规则：
- 文档更新：
```

## 17. 实验任务

1. 部署 SGLang，测量多轮对话场景的 prefix cache hit rate
2. 对比 SGLang vs vLLM 在 few-shot 场景下的 TTFT
3. 实现 JSON schema constrained decoding，测量 overhead
4. 测试不同 `--mem-fraction-static` 对缓存命中率的影响
5. 模拟 Radix Tree 容量不足时的性能退化

## 18. 习题 20 道

1. RadixAttention 的核心数据结构是什么？它如何实现自动前缀缓存？
2. SGLang 的 Radix Tree 与 vLLM 的 prefix caching 有什么区别？
3. Constrained decoding 的 FSM 是如何工作的？每步 decode 的额外开销是什么？
4. 什么是 PD Disaggregation？SGLang 如何实现？
5. Mooncake 的外部 KV Cache 解决了什么问题？
6. RBG 架构的优势和劣势是什么？
7. 如何计算 prefix cache hit rate？什么场景下 hit rate 最高？
8. Radix Tree 的 LRU 淘汰策略有什么问题？如何改进？
9. SGLang 的 chunked prefill 与 vLLM 的有什么区别？
10. 如何验证 constrained decoding 的正确性？
11. SGLang 的调度策略 "lpm" 是什么？为什么适合前缀缓存场景？
12. 在 Agent 场景中，SGLang 的哪些特性最有价值？
13. 如何估算 Radix Tree 的内存开销？
14. SGLang 支持哪些结构化输出格式？各自的实现机制？
15. 对比 SGLang 和 vLLM 在 throughput 和 latency 上的差异。
16. 什么情况下应该禁用 RadixAttention？
17. SGLang 的 overlap scheduling 是什么？如何提高 GPU 利用率？
18. 如何在 SGLang 中实现 multi-modal 推理？
19. SGLang 的 KV cache 量化支持情况如何？
20. 如何从 vLLM 迁移到 SGLang？需要注意什么？

## 19. 标准答案

1. Radix Tree（压缩前缀树）。每个节点存储一段 token 序列及其对应的 KV cache 物理地址。新请求到达时，沿树查找最长匹配前缀，复用已缓存的 KV cache。

2. vLLM 的 prefix caching 需要显式指定前缀（通过 hash），且粒度较粗。SGLang 的 RadixAttention 自动识别任意长度的公共前缀，粒度更细（token 级别），且支持动态增长。

3. FSM 预编译输出格式为状态转移图。每步 decode 时，根据当前 FSM 状态确定合法的下一个 token 集合，将非法 token 的 logit 设为 -inf。额外开销：FSM 状态查询 + token mask 应用，通常 < 0.1ms/token。

4. PD Disaggregation 将 prefill（计算密集）和 decode（内存密集）分离到不同 GPU 组。SGLang 通过 RBG 架构实现：prefill GPU 计算 KV cache → 传输到 decode GPU → decode GPU 执行生成。

5. Mooncake 将 KV cache 存储在独立的高速存储系统中（如 RDMA 网络连接的内存池），解决单 GPU 显存不足的问题，支持超长上下文和大规模并发。

(后续答案略)

## 20. 调优 checklist

- [ ] 确认 RadixAttention 已启用
- [ ] 监控 prefix cache hit rate > 目标值
- [ ] 设置合理的 `--mem-fraction-static`
- [ ] 验证 chat template 一致性（影响前缀匹配）
- [ ] 测试 constrained decoding 的 FSM 编译时间
- [ ] 配置 chunked prefill size 适配最大序列长度
- [ ] 设置 max-running-requests 防止 OOM
- [ ] 监控 eviction rate，必要时扩容
- [ ] 验证多轮对话的 TTFT 符合 SLA
- [ ] 压测确认 throughput 满足业务需求
