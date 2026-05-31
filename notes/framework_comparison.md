# vLLM / SGLang / TensorRT-LLM Comparison

This note compares three LLM serving stacks from the perspective of interviews and system design. The core split is:

- vLLM: flexible online serving built around continuous batching and PagedAttention.
- SGLang: serving plus programming/runtime features, especially RadixAttention prefix reuse and structured generation.
- TensorRT-LLM: compiled NVIDIA stack optimized through TensorRT engines, plugins, and runtime scheduling.

## Executive Comparison

| Dimension | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| scheduler | Step-level scheduler manages waiting, running, and swapped queues; prioritizes keeping running decode requests moving, then swapped, then new prefill. | Scheduler is prefix-aware; local materials call out `lpm` longest-prefix-match policy and overlap/chunked prefill for better prefix reuse. | Runtime scheduler supports inflight batching; policy examples include `guaranteed_no_evict` and `max_utilization`. |
| KV cache | PagedAttention stores KV cache in fixed-size physical blocks with logical-to-physical block tables and copy-on-write sharing. | RadixAttention stores reusable prefix KV in a radix tree with metadata, refcounts, and LRU-style eviction. | Paged KV cache is integrated with the compiled engine/runtime; build/runtime params define block size and total KV capacity. |
| prefix caching | Supported and useful for shared system prompts or multi-turn chat, but local material frames it as less automatic than SGLang. | First-class feature: automatic longest shared-prefix lookup and reuse at token/prefix granularity. | Possible through paged KV/context caching patterns, but less emphasized in local material than engine-level kernel/runtime optimization. |
| batching | Continuous batching keeps active decode sequences on GPU and admits/removes requests dynamically. Chunked prefill controls long prompt interference. | Batching is shaped by prefix sharing, structured generation constraints, and chunked prefill; prefix hits can turn large prefill into small prefill/decode work. | Inflight batching lets requests enter/leave active batches dynamically; compile-time limits such as max batch, max input, max seq, and max tokens constrain runtime behavior. |
| PD separation | Local vLLM material focuses on mixed prefill/decode scheduling and chunked prefill, not full disaggregation as the main design. | Local material explicitly ties SGLang to flexible PD separation, RBG, and external KV cache designs. | Can be part of a larger deployed system, but the local TensorRT-LLM note focuses more on single-runtime build/deploy optimization than PD architecture. |
| kernel/plugin strategy | Uses specialized kernels such as PagedAttention and can use CUDA Graph to reduce launch overhead. Flexible Python/PyTorch-oriented stack. | Uses runtime techniques around RadixAttention and constrained decoding; local material emphasizes scheduler/cache logic more than custom kernel compilation. | Core strength: TensorRT graph compilation, layer fusion, kernel selection, FMHA/context_fmha, GEMM and attention plugins, quantized kernels, custom all-reduce. |
| software stack | Python-facing serving engine with OpenAI-compatible API, AsyncLLMEngine, Workers, ModelRunner, PyTorch/CUDA kernels. | Serving/runtime plus Python DSL and structured generation; OpenAI-compatible usage is common, but local material emphasizes programmable generation and cache-aware scheduling. | HuggingFace checkpoint conversion, TensorRT-LLM model definition, `trtllm-build`, serialized engine, C++ runtime or Triton Inference Server. |
| deployment fit | Best when you need fast iteration, broad model support, high-throughput online APIs, long context, and good default serving behavior. | Best for multi-turn, few-shot, agent/tool calling, JSON/regex constrained output, and workloads with repeated prompt prefixes. | Best for fixed models on NVIDIA GPUs where maximum throughput/latency and quantization are worth build complexity. |
| risks | OOM from aggressive GPU memory utilization, preemption, queue starvation, TTFT spikes from long prefill, CUDA Graph memory/warmup overhead. | Cache thrashing, prefix mismatch from tokenization/template drift, hash/refcount correctness, FSM compile blowups, extra CPU/GPU metadata overhead. | Long builds, version/GPU architecture coupling, opaque debugging, compile-time shape limits, quantization quality loss, plugin/runtime compatibility risk. |

## Request Lifecycle And Mental Model

### vLLM

Typical request lifecycle:

1. Client sends an OpenAI-style request.
2. `AsyncLLMEngine.generate()` accepts it and places sequences into scheduler state.
3. The scheduler decides which sequence groups run this step, considering waiting, running, and swapped queues.
4. The block manager checks and allocates KV cache physical blocks.
5. Worker and ModelRunner prepare input tokens, positions, and block tables.
6. The model forward pass runs PagedAttention over KV blocks.
7. The sampler emits token candidates.
8. Scheduler state is updated, completed blocks are freed, and tokens stream back to the client.

Scheduler:

- Keeps three conceptual queues: waiting requests needing prefill, running requests in decode, and swapped/preempted requests.
- Prioritizes already-running decode work to avoid wasting allocated state and causing TPOT jitter.
- Uses limits such as max sequences, max batched tokens, preemption mode, and chunked prefill to balance throughput and latency.

Block manager:

- Treats KV cache like virtual memory.
- Maps logical token blocks for a sequence to physical GPU KV blocks.
- Supports copy-on-write, especially useful when beams or related sequences share prefix blocks.
- Frees blocks when sequences finish and may swap/recompute under pressure depending on configuration.

PagedAttention:

- Attention kernels read KV through block tables rather than assuming contiguous per-request KV memory.
- This reduces fragmentation and avoids preallocating maximum sequence length for every request.
- The interview answer: PagedAttention makes the KV cache allocation problem look like paged virtual memory, which improves GPU memory utilization and enables larger dynamic batches.

### SGLang

Request lifecycle:

1. Frontend receives a request and tokenizes it.
2. Runtime computes or looks up a prefix identity.
3. Radix tree lookup finds the longest cached prefix.
4. Scheduler decides whether the request needs full prefill, partial prefill, or mostly decode.
5. Runtime executes prefill/decode and inserts new KV segments into the radix tree.
6. Cache metadata, refcounts, and eviction policy decide what remains reusable.

RadixAttention:

- Stores prefix KV cache in a radix tree, where shared token prefixes map to reusable KV segments.
- A new request reuses the longest matching prefix and computes only the suffix.
- This directly reduces TTFT for multi-turn chat, few-shot prompts, shared system prompts, and prompt-template workloads.

Prefix sharing:

- vLLM also has prefix caching, but the local SGLang material frames SGLang's approach as more automatic and granular.
- Prefix sharing is only as good as stable tokenization and chat templates. Small formatting or tokenizer version changes can destroy hit rate.
- Hot prefixes can cause hot workers in distributed deployments, so routing must balance locality with queue pressure.

Structured generation fit:

- SGLang is a strong fit for JSON schema, regex-constrained output, tool calling, and agent workflows.
- Constrained decoding can be implemented by compiling the format constraint to an FSM, masking illegal tokens each decode step, then updating FSM state.
- The risk is per-token masking overhead and FSM state explosion for complex schemas or regexes.

Mooncake / external KV relation:

- Local materials connect SGLang to PD separation, RBG, and Mooncake-style external KV cache.
- The key design is KV-cache-centric scheduling: place or find KV near future decode instead of treating GPUs as interchangeable compute slots.
- External KV can help long context, multi-turn reuse, and PD disaggregation, but introduces KV transfer cost, metadata correctness, and locality/load-balance tradeoffs.

### TensorRT-LLM

Build/runtime split:

1. Start with HuggingFace model weights.
2. Convert checkpoints into TensorRT-LLM format.
3. Define/build the model network with TensorRT-LLM.
4. Run `trtllm-build` to produce a serialized TensorRT engine.
5. Deploy with the C++ runtime or Triton Inference Server.
6. Runtime handles tokenization integration, inflight batching, KV cache, sampling, and response streaming.

Inflight batching:

- Unlike static batching, requests can join and leave active batches as they arrive or finish.
- This improves GPU utilization for variable-length generation.
- Compile-time and runtime limits still matter: max batch size, max input length, max sequence length, max tokens, and KV cache capacity.

Plugins and kernels:

- TensorRT-LLM relies on graph optimization, layer fusion, kernel selection, memory planning, and precision calibration.
- Plugins extend TensorRT for LLM-specific operations such as attention variants, RoPE-like custom ops, GEMM choices, context FMHA, paged KV cache, quantization, and custom all-reduce.
- `context_fmha` is especially important for prefill because it fuses context attention work and avoids unnecessary global memory traffic.

Deployment tradeoffs:

- Strong when serving a stable model on NVIDIA data-center GPUs with strict performance or cost goals.
- Weak for rapid model iteration, non-NVIDIA targets, frequently changing model structures, and cases where compile time or engine version coupling is unacceptable.
- Engines are tied to build parameters, TensorRT/runtime versions, and GPU architecture assumptions; this is operationally powerful but less flexible.

## Prefill vs Decode Implications

Prefill:

- Processes all prompt tokens at once.
- Dominated by large GEMMs and attention over prompt length.
- Usually compute-bound.
- Drives TTFT.
- Benefits from FlashAttention/context FMHA, tensor parallelism, chunked prefill, and prefix cache hits.

Decode:

- Processes one new token per active sequence per step.
- Reads historical KV cache every step.
- Usually memory-bandwidth and scheduling sensitive.
- Drives TPOT/ITL and streaming smoothness.
- Benefits from larger continuous batches, GQA/MQA, KV cache quantization, CUDA Graph, paged KV layout, and speculative decoding.

Framework consequences:

- vLLM uses continuous batching to keep decode throughput high and chunked prefill to reduce prefill blocking.
- SGLang reduces prefill work by reusing prefix KV and is attractive when many requests share stable prefixes.
- TensorRT-LLM optimizes both phases through compiled kernels, but its build-time shape choices must match real input/output length distributions.
- PD separation is useful when prefill and decode resource ratios differ strongly, but KV transfer can erase the win for long prompts unless routing and transfer are engineered carefully.

KV transfer formula for PD separation:

```text
kv_transfer_bytes = prompt_tokens * 2 * layers * kv_heads * head_dim * bytes_per_element
```

The interview point: PD separation is not automatically faster. It trades prefill/decode isolation for queueing plus KV movement. It wins when independent scaling and better batching outweigh transfer latency and metadata complexity.

## Framework Interview Q&A

### vLLM

**Q1: What problem does PagedAttention solve?**  
A: It solves KV cache fragmentation and over-reservation. Instead of allocating one contiguous max-length KV region per request, vLLM stores KV in fixed-size blocks and uses block tables, similar to virtual memory paging.

**Q2: How does the vLLM scheduler think about requests?**  
A: It separates waiting prefill work, running decode work, and swapped/preempted work. Each step chooses a batch under token, sequence, and KV cache constraints.

**Q3: Why does continuous batching improve throughput?**  
A: Decode requests finish at different times. Continuous batching removes finished sequences and admits new work every step, keeping GPU work dense instead of waiting for a static batch to finish together.

**Q4: When should chunked prefill be enabled?**  
A: When long prompts block decode and cause TTFT/TPOT spikes. Chunked prefill splits large prompt processing so decode can continue making progress between chunks.

**Q5: What are common production risks in vLLM?**  
A: OOM from high memory utilization or long prompts, preemption jitter, waiting queue starvation, prefix cache memory pressure, and CUDA Graph warmup or memory overhead.

### SGLang

**Q1: What is RadixAttention?**  
A: It is a radix-tree-based prefix KV cache. The runtime finds the longest shared prefix for a request, reuses that KV cache, and computes only the unmatched suffix.

**Q2: Why is SGLang strong for multi-turn and few-shot workloads?**  
A: Those workloads reuse stable system prompts, histories, examples, or templates. RadixAttention can turn repeated prefill into prefix hits, reducing TTFT.

**Q3: How does constrained decoding work conceptually?**  
A: A schema or regex is compiled into an FSM. At each decode step, the runtime masks tokens that would violate the current FSM state, samples from legal tokens, and advances the FSM.

**Q4: How does Mooncake-style external KV change scheduling?**  
A: It makes KV cache placement a first-class scheduling object. Routing considers where reusable KV already lives, the cost to transfer it, and decode queue pressure.

**Q5: What can make SGLang prefix caching fail in practice?**  
A: Tokenizer/template drift, low prefix reuse, cache capacity pressure, LRU thrashing, hot-prefix worker imbalance, or bugs in hash/refcount/eviction metadata.

### TensorRT-LLM

**Q1: What is the build/runtime split?**  
A: Build converts and compiles a model into a TensorRT engine with fixed optimization assumptions. Runtime loads that engine and handles batching, KV cache, decoding, and serving integration.

**Q2: What is inflight batching?**  
A: It is dynamic batching where requests can enter and leave the active batch as they arrive or finish, unlike static batching where all requests start and end together.

**Q3: Why are plugins important in TensorRT-LLM?**  
A: TensorRT does not natively cover every LLM-specific operator or optimal kernel. Plugins provide optimized implementations for attention, GEMM variants, RoPE-like ops, paged KV cache, quantization, and communication.

**Q4: Why can TensorRT-LLM be harder to operate than vLLM?**  
A: Engines require build time, are coupled to TensorRT/GPU/version/shape choices, are less transparent to debug, and often need rebuilds for model or shape changes.

**Q5: When is TensorRT-LLM the best choice?**  
A: When the model is stable, hardware is NVIDIA, performance targets are strict, and the team can afford compile, validation, quantization, and rollout discipline.

## Deployment Heuristics

- Choose vLLM first for general online serving, iteration speed, OpenAI-compatible endpoints, and strong baseline throughput.
- Choose SGLang when prefix reuse and structured generation are central product requirements.
- Choose TensorRT-LLM when the deployment target is stable enough to justify engine compilation and NVIDIA-specific optimization.
- Consider PD separation when prefill and decode have different scaling bottlenecks, but always estimate KV transfer bytes and queueing first.
- Treat tokenizer version, model version, quantization, max context, and adapter identity as cache keys; otherwise prefix/KV reuse can become incorrect.

## Needs Source Verification

These claims are present or implied in local study material, but should be checked against upstream source or current docs later before using them as hard facts:

- Exact vLLM scheduler queue priority and current preemption behavior across recent versions.
- Current vLLM prefix caching implementation details, including cache key construction and eviction policy.
- Whether all local statements about SGLang `lpm`, overlap scheduling, PD separation, RBG, and Mooncake integration match current upstream behavior.
- Exact SGLang RadixAttention hash/refcount/eviction implementation and safeguards against prefix mismatch.
- Current TensorRT-LLM defaults for `paged_kv_cache`, `tokens_per_block`, `context_fmha`, inflight batching, and batch scheduler policies.
- Which TensorRT-LLM features are build-time only versus runtime configurable in the current release.
- Current support matrix for FP8, INT8/INT4, KV cache quantization, custom all-reduce, and non-Hopper/Ampere GPUs.
- Production claims about relative latency or throughput between vLLM, SGLang, and TensorRT-LLM; these require benchmark reproduction on the target model, hardware, and workload.
