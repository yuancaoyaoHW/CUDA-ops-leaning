# Hardware Selection for LLM Training and Inference

## Executive Summary

Hardware selection for LLM workloads is mainly a memory, interconnect, and software decision. Peak FLOPS matters, but it is rarely the first constraint once models are deployed with real request lengths, batching, KV cache, tensor parallelism, and operational SLOs.

For small 7B/14B latency-first inference, prefer the simplest single-card deployment that fits weights, KV cache, and runtime overhead with headroom. NVIDIA L40S, H100/H200 NVL, or similar CUDA-friendly cards are the easiest default when software velocity matters. L4 and Atlas 300I Pro are efficiency cards, but their smaller memory and lower bandwidth make them better for smaller models, quantized serving, or edge-style capacity pools.

For 32B/70B throughput-first inference, memory capacity and HBM bandwidth become first-order. H200, B200/B300, GB200/GB300, AMD MI300X/MI325X/MI350-series, and MI355X are stronger fits than smaller PCIe inference cards. NVIDIA has the strongest TensorRT-LLM/CUDA ecosystem; AMD has large HBM capacity and competitive bandwidth through ROCm when the model and serving stack are validated.

For long-context inference, choose capacity and bandwidth before raw compute. Larger HBM lets the system keep more KV cache resident; higher HBM bandwidth helps decode attention and other memory-bound kernels. H200's 141 GB, B200's 180 GB, MI300X's 192 GB, MI325X's 256 GB, and MI350/MI355X's 288 GB class are attractive single-accelerator data points from the matrix. Rack-scale systems such as GB200/GB300 NVL72 and Atlas 900 A3 shift the question from card selection to KV placement, routing, and interconnect topology.

For MoE inference and large model pretraining, interconnect dominates. Expert parallelism introduces all-to-all traffic; tensor/data parallel training introduces collectives such as all-reduce. Prefer NVLink/NVSwitch domains for NVIDIA, validated Infinity Fabric plus cluster networking for AMD, Ethernet/RoCE-oriented Gaudi clusters where the software stack is acceptable, or Ascend SuperPoD designs in localization-driven environments.

Domestic availability and localization can justify Huawei Ascend despite incomplete public chip-level specs. The trade is ecosystem maturity and portability: CUDA/TensorRT-LLM is the broadest path, ROCm is improving and capacity-rich, Gaudi emphasizes open Ethernet/RoCE scaling, and Ascend requires CANN/MindIE/MindSpore or Ascend PyTorch validation.

## Spec Comparison Summary

Use the hardware matrix as the source of truth. The table below compresses the public specs into selection implications rather than re-ranking every accelerator.

| Family | Matrix highlights | Best fit | Main caution |
| --- | --- | --- | --- |
| NVIDIA H100/H200 | H100 SXM: 80 GB HBM3, 3.35 TB/s; H200: 141 GB HBM3e, 4.8 TB/s; SXM NVLink 900 GB/s | Proven training/inference baseline, TensorRT-LLM, vLLM/SGLang on CUDA, latency-sensitive and high-throughput serving | H100 80 GB can be tight for long context or large batches; H200 improves capacity but still requires careful KV sizing |
| NVIDIA B200/B300 and DGX/HGX | B200: 180 GB, up to 8 TB/s; B300: 288 GB, up to 8 TB/s; HGX/DGX systems expose high aggregate NVLink bandwidth | Dense and MoE inference/training where FP4/FP8 and NVLink/NVSwitch matter | Standalone per-precision tables are incomplete in the captured matrix; use official system-level values carefully |
| NVIDIA GB200/GB300 NVL72 | GB200: 13.4 TB GPU HBM, 576 TB/s GPU memory bandwidth, 72-GPU NVLink domain; GB300: 20 TB GPU HBM, 576 TB/s, 800 Gb/s network connectivity per GPU | Rack-scale reasoning, trillion-parameter inference, MoE, frontier training | Liquid cooling, rack procurement, and operational maturity dominate TCO |
| NVIDIA L40S/L4 | L40S: 48 GB GDDR6 ECC, 864 GB/s, 350 W; L4: 24 GB GDDR6, 300 GB/s, 72 W | Efficient small-model inference, visual AI, lower-power capacity pools | GDDR bandwidth and smaller memory limit long-context and large-model decode throughput |
| AMD MI300X/MI325X | MI300X: 192 GB HBM3, 5.325 TB/s; MI325X: 256 GB HBM3e, 6 TB/s | Large-memory inference, fine-tuning, throughput serving, non-CUDA clusters | ROCm/model/runtime compatibility must be validated before committing production traffic |
| AMD MI350X/MI355X | MI350X/MI355X: 288 GB HBM3e, up to 8 TB/s; MI355X publishes MXFP4/MXFP6/MXFP8 and BF16/FP16 matrix values | FP4/FP6/FP8-era inference and training, high-capacity long-context serving | Matrix has unknowns for some platform interconnect and power fields |
| Intel Gaudi 3 | 128 GB HBM2e, 3.7 TB/s, Ethernet/RoCE-oriented scaling, PCIe card power noted as 600 W | Cost-sensitive training/inference clusters that value standard Ethernet fabrics | Full precision table was not captured as official; software stack and model coverage need proof |
| Huawei Ascend / Atlas | Atlas 800I A2 exposes 32 GB/64 GB NPU variants and 200GE RoCE; Atlas 900 A3 publishes 48 TB on-chip memory unified addressing and up to 384 Ascend 910C chips | Domestic/localized deployments, Ascend-native stacks, China-market supply constraints | 910B/910C chip-level HBM, bandwidth, precision, and power remain unknown in the matrix |

Metric effects:

- HBM capacity determines whether weights, KV cache, activations/workspace, communication buffers, and fragmentation headroom can fit. More HBM enables larger batch, longer context, more resident adapters, and fewer evictions.
- HBM bandwidth governs decode-heavy serving because decode attention repeatedly reads historical KV cache and many decode kernels have low arithmetic intensity. Higher bandwidth also helps memory-bound normalization, softmax, dequantization, and small-batch GEMV-like paths.
- BF16 remains the conservative training precision. FP8 reduces memory traffic and can improve training/inference throughput when kernels, scaling, and quality validation are mature. FP4 is mainly an inference and emerging training/reasoning lever; do not assume quality without evals. FP6/MXFP formats on AMD MI350-series should be treated as stack-specific until the workload is validated.
- Interconnect decides how painful TP, PP, EP, and DP become. TP needs frequent intra-layer communication, DP training needs all-reduce, PP needs activation transfer and scheduling balance, and EP/MoE needs all-to-all.
- Software stack affects time-to-production. CUDA and TensorRT-LLM have the broadest LLM path. ROCm can be strong on supported models and kernels but needs compatibility checks. Gaudi software and SynapseAI/PyTorch integration need cluster-specific proof. Ascend uses CANN, AscendCL, MindIE, MindSpore, and torch-npu/Ascend PyTorch paths.
- Power, cooling, and TCO can outweigh token/s. L4-class 72 W cards are operationally different from 700 W H100/H200 SXM, 750 W MI300X, 1000 W MI325X, and liquid-cooled NVL/SuperPoD systems.
- Single-card deployments optimize simplicity and failure isolation. Rack-scale systems optimize aggregate capacity, interconnect locality, and large model fit, but add scheduling, networking, cooling, and maintenance complexity.

## Recommendations by Workload

### 1. Single-node small-model inference: 7B/14B, small batch, latency first

Choose the simplest card that fits the quantized or BF16 model plus KV cache headroom. For latency-first work, avoid unnecessary tensor parallelism because inter-GPU synchronization can add TPOT jitter.

Recommended direction:

- First choice when CUDA compatibility matters: NVIDIA L40S for 48 GB class single-card inference, or H100/H200 NVL when stricter latency or larger context headroom is needed.
- Efficiency pools: NVIDIA L4 for smaller/quantized models and lower-power serving.
- Domestic/localized path: Atlas 300I Pro for smaller inference workloads where 24 GB LPDDR4X and CANN/MindIE support are enough.

Selection logic:

- HBM capacity decides whether the model and KV cache fit at the target context length.
- HBM bandwidth matters even at small batch because decode is often memory-bound.
- FP8/INT8/INT4 can reduce memory pressure, but use quality evals and avoid assuming peak TOPS translates to latency.
- Prefer one card over TP for 7B/14B unless context length, batch, or precision forces sharding.

### 2. Medium-model inference: 32B/70B, throughput first

Throughput serving wants large resident batches and stable decode steps. That points to high HBM capacity, high bandwidth, and fast same-node interconnect when tensor parallelism is needed.

Recommended direction:

- NVIDIA H200, B200/B300, DGX/HGX, or GB200/GB300 when TensorRT-LLM/CUDA optimization and NVLink/NVSwitch are the priority.
- AMD MI300X/MI325X/MI350-series/MI355X when large HBM per accelerator and ROCm compatibility are proven for the model stack.
- Intel Gaudi 3 where Ethernet/RoCE scaling and Gaudi software are already validated.

Selection logic:

- For 70B BF16-style deployments, single-card fit is often less important than replica shape: TP within a fast interconnect domain plus enough leftover HBM for KV cache.
- Higher HBM bandwidth improves decode token throughput and helps keep larger active batches useful.
- FP8 can be attractive for inference if the serving framework has optimized kernels and the model passes quality gates.
- Continuous batching, PagedAttention, and admission control matter as much as card specs.

### 3. Long-context inference: KV cache capacity/bandwidth first

Long context turns KV cache into the dominant resource. The relevant question is not only "can weights fit?" but "how many active tokens can remain in HBM without eviction or remote transfer?"

Recommended direction:

- Card/node class: H200, B200/B300, MI300X/MI325X/MI350X/MI355X for larger per-accelerator HBM.
- Rack class: GB200/GB300 NVL72 or Atlas 900 A3 when the workload needs a large interconnect domain and very large aggregate memory.
- Framework direction: vLLM PagedAttention, SGLang RadixAttention/prefix reuse, or TensorRT-LLM paged KV cache on NVIDIA.

Selection logic:

- HBM capacity controls maximum resident KV cache and therefore active sequence count, context length, and batch size.
- HBM bandwidth controls decode attention because each generated token reads historical K/V.
- KV cache quantization can double effective capacity in principle when moving from BF16 to FP8-sized storage, but quality and dequant overhead must be measured.
- PD separation can help, but KV transfer bytes scale with prompt tokens; locality-aware routing is mandatory for very long prompts.

### 4. MoE inference: expert routing, all-to-all, interconnect first

MoE reduces per-token compute by activating only selected experts, but it introduces expert routing, load imbalance, and all-to-all communication.

Recommended direction:

- NVIDIA GB200/GB300 NVL72 or HGX/DGX-class systems when a large NVLink/NVSwitch domain is available.
- AMD clusters only after validating expert parallel communication on the target ROCm/runtime stack.
- Ascend Atlas 900 A3/SuperPoD-style systems when localization and Ascend-native MoE support are the driver.

Selection logic:

- Interconnect is the first spec to inspect. Expert parallelism is only attractive if all-to-all latency and bandwidth are controlled.
- HBM capacity still matters because each worker may host multiple experts and KV cache.
- Load balancing metrics such as per-expert token count, routing skew, dropped tokens, and all-to-all latency should be deployment gates.
- Single-card specs are insufficient for MoE; evaluate the full node or rack topology.

### 5. LoRA/QLoRA fine-tuning: memory capacity and software ecosystem first

Fine-tuning is less communication-heavy than large-scale pretraining but still sensitive to memory, kernel coverage, optimizer support, and debugging tools.

Recommended direction:

- NVIDIA H100/H200/B200/B300 when CUDA, PyTorch, FlashAttention, quantization, and adapter tooling compatibility matter most.
- AMD MI300X/MI325X/MI350-series when ROCm support for the target training stack is confirmed and larger HBM reduces offload.
- Ascend when the organization is committed to CANN/MindSpore/torch-npu and has local support.

Selection logic:

- HBM capacity controls whether base weights, LoRA adapters, activations, optimizer states, and batch/sequence choices fit.
- BF16 is the conservative path for adapter training. QLoRA-style training shifts pressure toward quantization kernel support and optimizer/runtime compatibility.
- Single-node fine-tuning is usually easier to stabilize than multi-node training; prefer fewer moving parts unless capacity forces sharding.

### 6. Large model pretraining: multi-node interconnect, collective communication, stability first

Pretraining is a cluster problem. The accelerator is only one part of the decision; network, collectives, checkpointing, scheduler, thermal stability, and vendor support matter equally.

Recommended direction:

- NVIDIA DGX/HGX/GB200/GB300-class systems when the priority is the most mature CUDA/NCCL/training ecosystem and NVLink/NVSwitch locality.
- AMD MI300/MI325/MI350/MI355-class clusters when ROCm and distributed training have been proven at the intended scale.
- Intel Gaudi 3 where Ethernet/RoCE economics and Gaudi software are operationally validated.
- Huawei Atlas 900 A3/SuperPoD where domestic availability, CANN/MindSpore integration, and local support outweigh CUDA portability.

Selection logic:

- DP requires all-reduce; TP requires frequent intra-layer collectives; PP requires careful bubble management; EP requires all-to-all.
- HBM capacity affects microbatch size, activation checkpointing pressure, and model parallel shape.
- BF16/FP8 training precision depends on framework maturity, loss scaling/scaling-factor control, optimizer support, and recovery behavior.
- Stability, failure recovery, and checkpoint bandwidth are selection criteria, not afterthoughts.

### 7. Domestic availability / localization: Huawei Ascend vs NVIDIA/AMD/Intel tradeoffs

Ascend is strongest when supply chain, domestic procurement, and local software/control requirements dominate. NVIDIA is strongest for broad LLM software support. AMD is strongest when high HBM capacity and open GPU competition are valued and ROCm support is validated. Intel Gaudi is strongest when Ethernet/RoCE scaling and commercial availability align with the software stack.

Decision points:

- If the organization needs CUDA portability, TensorRT-LLM, and the broadest serving ecosystem, choose NVIDIA.
- If the organization needs large HBM per accelerator and can validate ROCm end to end, evaluate AMD seriously.
- If the organization wants Ethernet/RoCE-native scaling and can operate Gaudi software, evaluate Intel Gaudi 3.
- If localization is a hard requirement, evaluate Ascend with CANN, MindIE, MindSpore, and Ascend PyTorch, but keep chip-level unknowns explicit.

## NVIDIA vs AMD vs Huawei Ascend vs Intel Gaudi

### NVIDIA

Strengths:

- Widest production LLM ecosystem: CUDA, TensorRT-LLM, NCCL, Triton kernels, vLLM/SGLang support, NVIDIA AI Enterprise.
- Strong same-node interconnect story through NVLink/NVSwitch.
- Broad precision support across Hopper and Blackwell, including FP8 and FP4 on Blackwell-family systems.

Weaknesses:

- High acquisition cost and power/cooling requirements for top-end systems.
- Rack-scale systems require serious facility planning.
- Blackwell standalone peak tables are not fully captured in the matrix; use official system data and vendor docs, not derived claims.

### AMD

Strengths:

- Very large HBM per accelerator in MI300X, MI325X, MI350X, and MI355X.
- Strong HBM bandwidth in the matrix: MI300X 5.325 TB/s, MI325X 6 TB/s, MI350/MI355X up to 8 TB/s.
- ROCm and PyTorch support make AMD a practical alternative when validated.

Weaknesses:

- ROCm coverage, kernel availability, and serving framework maturity can vary by model and release.
- Some MI350-series platform interconnect and power fields are unknown in the captured matrix.
- Production teams need more up-front compatibility testing than with the common CUDA path.

### Huawei Ascend

Strengths:

- Domestic/localized deployment path with CANN, AscendCL, MindIE, MindSpore, and torch-npu/Ascend PyTorch.
- Atlas 800I/800T A2 and Atlas 900 A3 provide official system-level integration points.
- Atlas 900 A3 publishes large system-level memory/interconnect characteristics, including 48 TB on-chip memory unified addressing and up to 384 Ascend 910C chips.

Weaknesses:

- Public 910B/910C chip-level HBM capacity, HBM bandwidth, precision tables, and power remain unknown in the matrix.
- Porting CUDA-oriented models, kernels, and serving code requires engineering time.
- Third-party chip estimates should not be treated as official planning inputs.

### Intel Gaudi

Strengths:

- Gaudi 3 offers 128 GB HBM2e and 3.7 TB/s in the matrix.
- Ethernet/RoCE scaling can fit organizations that prefer open networking over proprietary same-node fabrics.
- PyTorch integration and Gaudi software provide a non-GPU alternative.

Weaknesses:

- Full official precision table was not captured in the matrix.
- Ecosystem depth is narrower than CUDA for many LLM serving paths.
- Cluster performance depends heavily on software maturity, networking, and model support.

## 10 Interview Q&A

**Q1: Why is HBM capacity often more important than peak FLOPS for LLM serving?**  
A: Serving must keep model weights, KV cache, workspace, communication buffers, and runtime headroom resident. If KV cache does not fit, the system evicts, swaps, rejects, or lowers batch size before peak FLOPS matters.

**Q2: Why does HBM bandwidth matter so much for decode?**  
A: Decode processes one new token per active sequence and repeatedly reads historical KV cache. That makes decode attention and many small-batch kernels memory-bound, so higher bandwidth can improve TPOT and throughput.

**Q3: How does KV cache change hardware selection for long context?**  
A: KV cache grows with layers, KV heads, head dimension, token count, and bytes per element. Long context therefore needs more HBM capacity and bandwidth, plus paged allocation and locality-aware routing.

**Q4: When is FP8 a good choice?**  
A: FP8 is useful when the accelerator, kernels, framework, scaling policy, and quality evals are mature. It can reduce memory traffic and improve Tensor Core utilization, but it is not a free replacement for BF16.

**Q5: What is the practical difference between BF16, FP8, and FP4?**  
A: BF16 is the conservative training and fine-tuning baseline. FP8 is a mature low-precision direction on modern training/inference stacks when validated. FP4 is mainly attractive for newer inference/reasoning stacks and requires stricter quality validation.

**Q6: Why does MoE care more about interconnect than dense inference?**  
A: MoE routes tokens to experts that may live on different accelerators. Expert parallelism creates all-to-all traffic, so routing skew and interconnect bandwidth/latency can dominate.

**Q7: Why is tensor parallelism easier inside one NVLink/NVSwitch domain?**  
A: TP communicates inside layers, often on every transformer block. Keeping that traffic on fast same-node interconnect reduces TPOT jitter compared with slower cross-node paths.

**Q8: When should a team choose AMD over NVIDIA?**  
A: AMD is attractive when large HBM capacity, bandwidth, pricing, or vendor strategy matter and the ROCm serving/training stack is validated for the exact models and frameworks.

**Q9: When does Ascend make sense despite incomplete public chip specs?**  
A: Ascend makes sense when localization, domestic supply, and Ascend-native software support are hard requirements. The deployment plan should rely on official Atlas system specs and internal validation rather than unofficial chip estimates.

**Q10: Why is rack-scale not just "more single cards"?**  
A: Rack-scale systems add large interconnect domains, shared cooling/power constraints, failure domains, scheduler topology, and networked collective behavior. The unit of selection becomes the rack or pod, not the accelerator card.

## Risks and Unpublished-Parameter Caveats

- Do not infer unknown fields. The matrix intentionally marks many Ascend 910B/910C chip-level values as unknown, including HBM capacity, memory bandwidth, precision table, sparsity behavior, and power.
- Do not treat third-party estimates as official facts. They may mix chip, module, card, dense, sparse, or system-level values.
- NVIDIA Blackwell and Blackwell Ultra rows are stronger at system-level specs than standalone per-GPU precision tables in the captured sources.
- Intel Gaudi 3 memory, bandwidth, form factor, Ethernet/RoCE positioning, and PCIe-card power are captured, but the full official precision table was not copied into the matrix.
- AMD MI350X has official memory/bandwidth coverage in the matrix, but some detailed platform and per-precision fields are incomplete except where MI355X publishes them.
- Published peak compute does not predict real LLM throughput by itself. Kernel coverage, batching, KV cache layout, quantization, interconnect, and scheduler policy decide production behavior.
- Power and cooling constraints can invalidate a nominally cheaper option. Compare watts, rack density, liquid-cooling requirements, facility capacity, and operations staffing.
- Single-card tests do not prove multi-node training. Multi-node selection must include collective communication, failure recovery, checkpointing, and thermal stability.
- Software lock-in is real. CUDA/TensorRT-LLM, ROCm, Gaudi software, and CANN/MindIE/MindSpore/Ascend PyTorch each imply different model conversion, kernel, debugging, and hiring costs.
- Always run workload-specific validation before procurement: target model, context distribution, batch policy, precision, adapter usage, serving framework, and cluster topology.
