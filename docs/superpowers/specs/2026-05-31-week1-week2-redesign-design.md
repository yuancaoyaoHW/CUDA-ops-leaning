# Week 1 / Week 2 重设计：Reduction-Normalization-Fusion 主题块

- **日期**: 2026-05-31
- **背景**: 现行 56 天计划 W1 用 5 个工作日只产出 `row_softmax` 一个算子，W2 用 7 天只产出 `rmsnorm` + 最小 CUDA extension。`progress.yaml` 显示 `vector_add` / `axpy` / `row_sum` / `row_max` 已经推到 benchmark/profile 阶段，Triton "开壳"成本基本消失，按一个算子一天的节奏继续走会浪费宝贵的前两周。
- **目标**: 把 W1/W2 重构成两个主题块，提升节奏（A）、深度（B）、横向串联（C）、复盘输出（D），并对标 Triton 官方 tutorials 做风格基线和源码走读。

## 设计原则

1. **算子总量从 1.5 → 约 10 个**，全部是 LLM 推理主流 op（softmax / masked / online / RMSNorm / LayerNorm / SwiGLU / GELU / RoPE / fused-add-norm / KV append）。
2. **每天三槽结构**：
   - **主线槽（3-4h）**：本日 kernel 闭环（reference → impl → tests → bench）。
   - **深度槽（1.5h）**：variant 对比 / tile sweep / numerical stability / profile 结论，强制产出可贴进面试的素材。
   - **输出槽（45min）**：源码走读或 Triton 官方 tutorial 对标，落到笔记。
3. **精心选 hook 给后周铺路**：Day 4 `online_softmax` → W4 FlashAttention；Day 10 `rope` → W4 attention；Day 13 `kv_cache_append` → W5 PagedAttention。
4. **对标 Triton 官方 tutorials**：每个对标日 git clone Triton repo 跑官方版本，把官方 ncu 数据和自己 variant 放同一张表，源码走读 / benchmark / 面经叙事一次完成。
5. **6 项算子成熟度仍是验收标准**（reference / implementation / tests / benchmark / profile / note）。
6. **Buffer 放在周检日**（Day 7 / Day 14），延迟任务消化或加深最薄弱环节。

## Triton 官方 tutorials 集成

参考 https://triton-lang.org/main/getting-started/tutorials ：

| Tutorial | 标题 | 落点 |
|----------|------|------|
| 01 | Vector Addition | Day 1 重读做风格基线 |
| 02 | Fused Softmax | Day 2-4（fused / autotune / ncu 三段拆开） |
| 03 | Matrix Multiplication | W3 |
| 04 | Low-Memory Dropout | Day 13 可选阅读，不强制 |
| 05 | Layer Normalization | Day 5-6（forward + backward 阅读） |
| 06 | Fused Attention | W4 |
| 07 | Libdevice | Day 9（SwiGLU/GELU 落地） |
| 08-10 | Group / Persistent / Block Scaled Matmul | W3 / W7 |

## W1：Reduction & Normalization 家族

| Day | 主线槽（3-4h） | 深度槽（1.5h） | 输出槽 / 对标 tutorial（45min） |
|-----|---------------|----------------|--------------------------------|
| 1 | 校准仓库 + Nsight Compute WSL2 验证 | 重新跑通已有 `row_max`/`row_sum` ncu | 重读 Tutorial 01 + 算子成熟度审计表 |
| 2 | `softmax` baseline + tests（复用 row_max/row_sum） | tile size sweep + GB/s 表 | Tutorial 02 第 1 段 + PyTorch `SoftMax.cu` 走读 |
| 3 | `masked_softmax` | warp shuffle vs shared-mem reduction 对比 | Tutorial 02 第 2 段（autotune + heuristics）+ vLLM softmax 走读 |
| 4 | `online_softmax` 推导 + 实现（→ W4 FlashAttention 铺路） | numerical stability：fp16 overflow / bf16 accumulate | Tutorial 02 ncu 段 + 面经 5 道（数值稳定 / launch-bound / occupancy） |
| 5 | `rmsnorm` baseline + tests + bench | vectorized load (`tl.load` 4xfp16 pack) | Tutorial 05 forward 段（Welford 均值/方差） |
| 6 | `layernorm` + 与 RMSNorm benchmark 对比 | 双 ncu profile 给 bottleneck 结论 | Tutorial 05 backward 段（locked atomic add，仅阅读）+ vLLM `layernorm.cu` 走读 |
| 7 | Week 1 周检 + STAR | TopK / RAII drill | reduction/norm 家族对比表（GB/s / arithmetic intensity / bottleneck） |

**W1 产出**：5 个新 kernel（softmax / masked / online / rmsnorm / layernorm）+ 4 段源码走读 + 1 张家族对比表 + 5+ 道面试题 + 1 段 STAR。

## W2：Fusion & Extension

| Day | 主线槽（3-4h） | 深度槽（1.5h） | 输出槽 / 对标 tutorial（45min） |
|-----|---------------|----------------|--------------------------------|
| 8 | `fused_add_rmsnorm`（residual + norm fusion） | fusion 前后 ncu 对比、launch overhead 量化 | vLLM `fused_add_rms_norm` 走读 |
| 9 | `swiglu` + `gelu` 激活 fusion | 双 gate matmul + activation fusion 思路 | Tutorial 07 Libdevice 完整阅读（`tanh`/`erf`/`rsqrt`/`silu`）+ fusion tradeoff 面经题 |
| 10 | `rope` 旋转位置编码（→ W4 attention 铺路） | in-place vs out-of-place、fp32 sin/cos cache | vLLM `pos_encoding.cu` 走读 |
| 11 | 最小 PyTorch C++/CUDA extension（推荐用 RMSNorm 重写一版） | setup.py vs JIT load、`TORCH_CHECK` | extension build/run/verify note |
| 12 | extension pytest + 与 Triton 版 benchmark | C++ template / dtype dispatch | C++ drill：smart pointer / RAII |
| 13 | `kv_cache_append` toy（→ W5 PagedAttention 铺路） | contiguous vs paged 写入差异（先做 contiguous） | 可选 Tutorial 04 Dropout（PRNG/seeded mask）+ 面经 5 道（CUDA extension / fusion / KV layout） |
| 14 | Week 2 阶段检 + STAR | LRU / smart pointer drill | reduction + norm + fusion + extension 全景表 + STAR 段 |

**W2 产出**：5 个新 kernel/扩展（fused-add-rmsnorm / swiglu / gelu / rope / kv_cache_append）+ 1 个可 pytest 的 CUDA extension + 3 段源码走读 + 全景表 + 1 段 STAR。

## 跨周衔接（hook）

| 算子 | 落地日 | 后续周复用 |
|------|--------|------------|
| `online_softmax` | Day 4 | W4 Day 23-24 FlashAttention online update 推导直接复用 |
| `rope` | Day 10 | W4 Day 22 attention reference 拼装时直接接入 |
| `kv_cache_append` (contiguous) | Day 13 | W5 Day 29-30 paged KV layout 在此基础上做 block table 抽象 |
| reduction/norm 家族对比表 | Day 7 / Day 14 | W3 GEMM roofline、W4 attention bottleneck 复用同一张分析模板 |

## 验收标准（与 README 6 项成熟度对齐）

每个新 kernel 必须达到：reference + implementation + tests + benchmark；其中至少 6 个 kernel 还要有 ncu profile + standardized note。两周末（Day 7、Day 14）必须存在两段 STAR（含 baseline / metric / improvement / failure / badcase 字段）。

## 风险与对策

| 风险 | 对策 |
|------|------|
| Day 4 online_softmax + Day 10 RoPE 数学密度大，可能侵占深度槽 | 允许这两天把深度槽并入主线，但输出槽（源码走读）不可砍 |
| Day 11 CUDA extension build path 在 WSL2 + CUDA 12.6 下踩坑 | Day 1 ncu 验证时同步验证 `torch.utils.cpp_extension` 能 build 通过；如失败，Day 11 主线降级为"读 vLLM 现成 extension 代码 + 解释 build 流程" |
| 节奏紧导致某天闭环不到 benchmark | Day 7 / Day 14 周检日的 buffer 时间用于补课，不顺延到下一周 |
| 官方 Triton repo 跑官方 tutorial 时硬件不匹配（4060 8GB） | 把 tutorial 的 problem size 缩小，只对标 ncu 指标趋势，不对标绝对数 |

## 不做什么（YAGNI）

- 不做 dropout kernel 落地（推理主线无关，仅 Day 13 可选阅读 Tutorial 04）。
- 不在 W1/W2 落地 backward kernel（仅阅读 Tutorial 05 backward 段，理解 atomic 用法即可）。
- 不在 W2 完整实现 FlashAttention（留 W4），仅做 online softmax 数学和 RoPE 这两块前置。
- 不重写 `vector_add` / `axpy` / `row_sum` / `row_max`，它们已达 benchmark/profile 阶段，仅在 Day 1 复跑用于 ncu 校准。

## 后续步骤

1. 用户复核本 spec。
2. 调用 writing-plans skill 把本设计转成实施计划（更新 `study-plan/inference-acceleration-plan.md` Day 1-14 表格，更新 `progress.yaml` 的算子条目，并把每日三槽落到 `study-plan/run.py` 信息屏的输出格式里）。
