# W1/W2 重设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec 把 W1/W2 改造成"reduction-normalization-fusion"主题块——把 `progress.yaml` 的 operators/week1/week2 段重写、`inference-acceleration-plan.md` Day 1-14 表替换、`run.py day N` 信息屏新增三槽（main/depth/output）渲染。

**Architecture:** 三块平行改动。(1) `progress.yaml` 加 9 个 operator stub + 重写 14 个 day 条目（新增 `slots:` 字段记录主线/深度/输出三槽）。(2) `inference-acceleration-plan.md` Day 1-14 表格替换 + change-log 追加。(3) `study-plan/run.py` 的 `_day_show` 新增 slots 渲染（TDD，改 fixture + 加测试）。schema 变更只在 W1/W2 落地，W3-W8 保持现状。

**Tech Stack:** Python 3.11、PyYAML、pytest（fixture-driven），无新增依赖。

**Spec:** `docs/superpowers/specs/2026-05-31-week1-week2-redesign-design.md`（commit `37faae0`）。

---

## 关键命名约定（lock-down）

`study-plan/verify.py` 的 `resolve_paths` 推断如下默认路径，本计划使用的算子命名严格遵守：

| 字段 | 约定 |
|---|---|
| impl | `kernels/triton/<op>/<op>.py` |
| tests | `tests/test_<op>.py` |
| bench | `benchmarks/bench_<op>.py` |
| bench_json | `reports/json/<op>_bench.json` |
| note | `notes/<op>.md` |
| profile | `reports/ncu/<op>*.ncu-rep` 或 `reports/nsys/bench_<op>*.nsys-rep` |

W1/W2 共涉及 11 个 operator（2 个已存在，9 个新增 stub）：

| operator | 状态 | kind | 默认路径 OK？ | 备注 |
|---|---|---|---|---|
| `row_softmax` | 已有 stub | reduction | ✓ | Day 2 |
| `masked_softmax` | **新增** | reduction | ✓ | Day 3 |
| `online_softmax` | **新增** | reduction | ✓ | Day 4 |
| `rmsnorm` | 已有 stub | reduction | ✓ | Day 5 |
| `layernorm` | **新增** | reduction | ✓ | Day 6 |
| `fused_add_rmsnorm` | **新增** | reduction | ✓ | Day 8 |
| `swiglu` | **新增** | elementwise | ✓ | Day 9 主算子 |
| `gelu` | **新增** | elementwise | ✓ | Day 9 副算子 |
| `rope` | **新增** | elementwise | ✓ | Day 10 |
| `rmsnorm_cuda_ext` | **新增** | reduction | ✗ 需 paths overrides | Day 11-12 |
| `kv_cache_append` | **新增** | elementwise | ✓ | Day 13 |

`rmsnorm_cuda_ext` 因为是 CUDA extension（不是 Triton kernel），需在 yaml 里显式写 `paths:` overrides 指向 `kernels/cuda/rmsnorm_cuda_ext/`。

---

## File Structure

| 路径 | 责任 | 操作 |
|---|---|---|
| `study-plan/progress.yaml` | 加 9 个 operator stub；重写 week1/week2 共 14 个 day 条目，每个 day 增加 `slots:` 三槽 | 修改 |
| `study-plan/inference-acceleration-plan.md` | Day 1-14 表替换；change-log 追加 W1/W2 重设计条目 | 修改 |
| `study-plan/run.py` | `_day_show` 在 artifacts 之后渲染 `slots:` 三槽 | 修改 |
| `tests/fixtures/study_plan/mini_progress.yaml` | `week1.day01` 加 `slots:` 段供测试断言 | 修改 |
| `tests/test_run_driver.py` | 新增 `test_day_show_renders_slots_when_present` | 修改 |
| `docs/superpowers/specs/2026-05-31-week1-week2-redesign-design.md` | 已存在，无需改 | 引用 |

**不动的**：W0 / W3-W8 yaml 段；`verify.py`（默认路径已能覆盖新算子，仅 `rmsnorm_cuda_ext` 用 `paths:` overrides）；`progress.py`；dashboard 相关；既有算子（`vector_add` / `axpy` / `row_sum` / `row_max`）的 yaml 条目。

---

### Task 1: `_day_show` 渲染 slots 字段（TDD）

**Files:**
- Modify: `tests/fixtures/study_plan/mini_progress.yaml:53-67`（在 `week1.day01` 加 `slots:` 段）
- Test: `tests/test_run_driver.py`（新增一个测试）
- Modify: `study-plan/run.py:92-156`（`_day_show` 函数末尾追加 slots 渲染）

- [ ] **Step 1: 在 fixture 的 day01 加 slots，预先匹配后续 yaml 改动**

打开 `tests/fixtures/study_plan/mini_progress.yaml`，找到 `week1.day01` 块（约第 53-67 行），在 `notes: ''` 一行后追加：

```yaml
    slots:
      main: '主线 fixture'
      depth: '深度 fixture'
      output: '输出 fixture'
```

注意缩进与 day01 内字段保持一致（4 空格 indent）。

- [ ] **Step 2: 在 `tests/test_run_driver.py` 末尾新增 slots 渲染测试**

```python
def test_day_show_renders_slots_when_present(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["day", "1"], cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "main:" in out and "主线 fixture" in out
    assert "depth:" in out and "深度 fixture" in out
    assert "output:" in out and "输出 fixture" in out
```

- [ ] **Step 3: 运行新测试确认它失败**

```bash
cd /home/ycy/code/llm-kernel-lab && python -m pytest tests/test_run_driver.py::test_day_show_renders_slots_when_present -v
```
Expected: FAIL（断言 `"main:"` 不在 out 中）。

- [ ] **Step 4: 在 `_day_show` 末尾追加 slots 渲染**

打开 `study-plan/run.py`，找到 `_day_show` 函数（约第 92 行起），在函数最后一段（`cmd = _default_command(...)` 那块）**之前**插入：

```python
    slots = day.get("slots") or {}
    if slots:
        print("  slots:")
        for key in ("main", "depth", "output"):
            if key in slots:
                print(f"    {key}: {slots[key]}")
```

完整定位：插入位置在 `# Show suggested command` 注释行之前。

- [ ] **Step 5: 跑新测试 + 全量 run_driver 测试**

```bash
cd /home/ycy/code/llm-kernel-lab && python -m pytest tests/test_run_driver.py -v
```
Expected: 全部 PASS（包含新增的 `test_day_show_renders_slots_when_present` 与原有 7 个测试）。

- [ ] **Step 6: 跑全套 study-plan 相关测试（fixture 改动可能波及）**

```bash
cd /home/ycy/code/llm-kernel-lab && python -m pytest tests/test_verify.py tests/test_progress_cli.py tests/test_study_plan_dashboard.py tests/test_run_driver.py tests/test_verify_drill.py -v
```
Expected: 全部 PASS。

- [ ] **Step 7: Commit**

```bash
cd /home/ycy/code/llm-kernel-lab && \
git add study-plan/run.py tests/test_run_driver.py tests/fixtures/study_plan/mini_progress.yaml && \
git commit -m "feat(run.py): render daily slots (main/depth/output) in day show"
```

---

### Task 2: 在 `progress.yaml` 加 9 个新 operator stub

**Files:**
- Modify: `study-plan/progress.yaml:45-129`（`operators:` 段，紧跟现有 `int4_dequant` 之前/之后插入）

- [ ] **Step 1: 打开 `study-plan/progress.yaml`，定位到 `operators:` 段末（在 `int4_dequant:` 块后、`gpu_libraries:` 行前）**

确认该位置（应在第 129 行附近，`flash_attention_toy` 之后、`int4_dequant` 之后、`gpu_libraries:` 之前）。

- [ ] **Step 2: 在 `int4_dequant` 块之后、`gpu_libraries:` 之前，插入以下 9 个 operator stub**

```yaml
  masked_softmax:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: reduction
  online_softmax:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: reduction
  layernorm:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: reduction
  fused_add_rmsnorm:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: reduction
  swiglu:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: elementwise
  gelu:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: elementwise
  rope:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: elementwise
  rmsnorm_cuda_ext:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: reduction
    paths:
      impl: kernels/cuda/rmsnorm_cuda_ext/rmsnorm_cuda_ext.cu
      tests: tests/test_rmsnorm_cuda_ext.py
      bench: benchmarks/bench_rmsnorm_cuda_ext.py
      bench_json: reports/json/rmsnorm_cuda_ext_bench.json
      note: notes/rmsnorm_cuda_ext.md
  kv_cache_append:
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    kind: elementwise
```

- [ ] **Step 3: 验证 yaml 解析正确 + verify 引擎不报错**

```bash
cd /home/ycy/code/llm-kernel-lab && \
python -c "import yaml; data=yaml.safe_load(open('study-plan/progress.yaml')); print(sorted([k for k in data['operators']]))"
```
Expected: 输出列表包含 `'fused_add_rmsnorm'`, `'gelu'`, `'kv_cache_append'`, `'layernorm'`, `'masked_softmax'`, `'online_softmax'`, `'rmsnorm_cuda_ext'`, `'rope'`, `'swiglu'`（以及原有 `'axpy'`, `'flash_attention_toy'`, `'int4_dequant'`, `'rmsnorm'`, `'row_max'`, `'row_softmax'`, `'row_sum'`, `'vector_add'`）。

```bash
cd /home/ycy/code/llm-kernel-lab && python study-plan/progress.py verify --operator masked_softmax 2>&1 | head -5
```
Expected: 输出 verify 结果（artifacts 全 false，无 traceback）。

- [ ] **Step 4: Commit**

```bash
cd /home/ycy/code/llm-kernel-lab && \
git add study-plan/progress.yaml && \
git commit -m "feat(progress): add 9 operator stubs for W1/W2 redesign"
```

---

### Task 3: 重写 `progress.yaml` 的 `week1` 段

**Files:**
- Modify: `study-plan/progress.yaml`（`week1:` 段，原 day01-day07，约第 183-327 行）

- [ ] **Step 1: 打开 `study-plan/progress.yaml`，定位 `week1:` 段（约第 183 行），把整段从 `week1:` 行起到 `week2:` 行前一整块替换为下面内容**

```yaml
week1:
  day01:
    title: 校准 + Nsight Compute WSL2 验证（warmup gate）
    date: '2026-06-01'
    status: not_started
    daily_check: 0
    jd_tags:
    - perf
    - docs
    tasks:
      audit_existing_kernels: false
      validate_ncu_wsl2: false
      reread_tutorial_01: false
    artifacts:
      docs: false
    verification: ''
    weaknesses: GPU 加速库、算子能力薄弱；当前进度记录和真实产物不同步。
    next_fix: 先确认 ncu 能采集硬件计数器，再进入 reduction-normalization 主题块。
    notes: ''
    phase: review
    slots:
      main: '校准仓库 + Nsight Compute WSL2 验证（重跑已有 row_max/row_sum 一次以校准）'
      depth: '跑通 row_max/row_sum 的 ncu，确认 memory-bound 结论可复现'
      output: '重读 Triton Tutorial 01（vector add）作为风格基线 + 算子成熟度审计表'
  day02:
    title: row_softmax baseline + tests + bench（reduction 主题开篇）
    date: '2026-06-02'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      write_reference: false
      implement_softmax: false
      add_tests: false
      add_benchmark: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: row_softmax
    phase: benchmark
    slots:
      main: 'row_softmax baseline + tests（复用 row_max/row_sum 的 reduction 砖块）+ bench'
      depth: 'tile size sweep（BLOCK_SIZE 256/512/1024/2048）→ 出 GB/s 表'
      output: 'Triton Tutorial 02 §1（naive vs fused）+ PyTorch SoftMax.cu 走读'
  day03:
    title: masked_softmax（attention mask 应用 + reduction 风格对比）
    date: '2026-06-03'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      implement_masked_softmax: false
      add_tests: false
      add_benchmark: false
      reduction_strategy_compare: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: masked_softmax
    phase: benchmark
    slots:
      main: 'masked_softmax 实现 + tests（causal mask + padding mask）+ bench'
      depth: 'warp shuffle vs shared-mem reduction 两种实现路径对比 GB/s'
      output: 'Triton Tutorial 02 §2（autotune + heuristics）+ vLLM softmax 走读'
  day04:
    title: online_softmax（→ W4 FlashAttention 铺路）
    date: '2026-06-04'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      derive_online_softmax_math: false
      implement_online_softmax: false
      add_tests: false
      add_benchmark: false
      add_profile_note: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: online_softmax
    phase: profile
    slots:
      main: 'online softmax 数学推导 + 单 pass 实现 + tests + bench + ncu profile'
      depth: 'numerical stability：fp16 overflow / bf16 accumulate 对比 + max-rescale 误差分析'
      output: 'Triton Tutorial 02 ncu 段 + 5 道面经题（数值稳定 / launch-bound 判别 / occupancy）'
  day05:
    title: rmsnorm baseline + tests + bench（normalization 主题开篇）
    date: '2026-06-05'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      write_reference: false
      implement_rmsnorm: false
      add_tests: false
      add_benchmark: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: rmsnorm
    phase: benchmark
    slots:
      main: 'rmsnorm reference + Triton 实现 + tests + bench'
      depth: 'vectorized load（tl.load 4xfp16 pack）vs naive load 对比 GB/s'
      output: 'Triton Tutorial 05 forward 段（Welford 均值/方差）走读笔记'
  day06:
    title: layernorm + 与 RMSNorm 对比（normalization 收口）
    date: '2026-06-06'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      implement_layernorm: false
      add_tests: false
      add_benchmark: false
      add_profile_note: false
      compare_rmsnorm: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: layernorm
    phase: profile
    slots:
      main: 'layernorm 实现 + tests + bench + ncu profile'
      depth: '双 ncu profile（rmsnorm vs layernorm）→ bottleneck 与 compute/memory ratio 对比表'
      output: 'Triton Tutorial 05 backward 段（locked atomic add）阅读笔记 + vLLM layernorm.cu 走读'
  day07:
    title: Week 1 周检 + STAR + reduction-norm 家族对比表
    date: '2026-06-07'
    status: not_started
    daily_check: 0
    weekly_check_score: 0
    jd_tags:
    - interview
    tasks:
      handwrite_softmax_oral: false
      family_compare_table: false
      topk_heap_drill: false
      cpp_raii_move_review: false
      star_metrics_entry: false
    artifacts:
      mock: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    phase: review
    star_filled: false
    algo_drill_done: false
    cpp_drill_done: false
    slots:
      main: '口头闭卷 mock：softmax / online softmax / rmsnorm / layernorm 推导'
      depth: 'reduction-normalization 家族对比表（GB/s / arithmetic intensity / bottleneck）'
      output: 'TopK / RAII drill + STAR 段（baseline/metric/improvement/failure/badcase）'
```

- [ ] **Step 2: 验证 yaml 仍可解析**

```bash
cd /home/ycy/code/llm-kernel-lab && python -c "import yaml; data=yaml.safe_load(open('study-plan/progress.yaml')); ws=data['week1']; print('day01 op:', ws['day01'].get('operator','(none)')); print('day02 op:', ws['day02'].get('operator')); print('day04 op:', ws['day04'].get('operator')); print('day06 op:', ws['day06'].get('operator')); print('day02 slots keys:', sorted(ws['day02']['slots'].keys()))"
```
Expected: `day01 op: (none)`、`day02 op: row_softmax`、`day04 op: online_softmax`、`day06 op: layernorm`、`day02 slots keys: ['depth', 'main', 'output']`。

- [ ] **Step 3: 跑 driver 信息屏 smoke test（用真实 yaml 而非 fixture）**

```bash
cd /home/ycy/code/llm-kernel-lab && python study-plan/run.py day 2 2>&1 | head -30
```
Expected: 输出包含 `Day 2:`、`operator: row_softmax`、`slots:`、`main:`、`depth:`、`output:`。

- [ ] **Step 4: Commit**

```bash
cd /home/ycy/code/llm-kernel-lab && \
git add study-plan/progress.yaml && \
git commit -m "feat(progress): rewrite week1 days as reduction-normalization theme block"
```

---

### Task 4: 重写 `progress.yaml` 的 `week2` 段

**Files:**
- Modify: `study-plan/progress.yaml`（`week2:` 段，原 day08-day14）

- [ ] **Step 1: 打开 `study-plan/progress.yaml`，定位到 `week2:` 行起、`week3:` 行前的整段，替换为下面内容**

```yaml
week2:
  day08:
    title: fused_add_rmsnorm（fusion 主题开篇 + W4 attention 前置）
    date: '2026-06-08'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      implement_fusion: false
      add_tests: false
      add_benchmark: false
      add_profile_note: false
      compare_unfused: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: fused_add_rmsnorm
    phase: profile
    slots:
      main: 'residual + rmsnorm fusion 实现 + tests + bench + ncu profile'
      depth: 'fusion 前后 ncu 对比 + launch overhead 量化（kernel 数 / launch 时间）'
      output: 'vLLM fused_add_rms_norm 走读笔记'
  day09:
    title: swiglu + gelu（激活 fusion + Libdevice 落地）
    date: '2026-06-09'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      implement_swiglu: false
      implement_gelu: false
      add_tests: false
      add_benchmark: false
      libdevice_intrinsics: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: swiglu
    phase: benchmark
    slots:
      main: 'swiglu（双 gate matmul + SiLU fusion）+ gelu 实现 + tests + bench'
      depth: 'fusion tradeoff（kernel launch / cache reuse / register pressure）面经题 5 道'
      output: 'Triton Tutorial 07 Libdevice 完整阅读（tanh/erf/rsqrt/silu）+ vLLM activation 走读'
  day10:
    title: rope（→ W4 attention 铺路）
    date: '2026-06-10'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      implement_rope: false
      add_tests: false
      add_benchmark: false
      sin_cos_cache_strategy: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: rope
    phase: benchmark
    slots:
      main: 'RoPE Triton 实现（in-place + 半精度 query/key）+ tests + bench'
      depth: 'in-place vs out-of-place、fp32 sin/cos cache vs 在线计算 对比'
      output: 'vLLM pos_encoding.cu 走读笔记'
  day11:
    title: 最小 PyTorch C++/CUDA extension（rmsnorm 重写一版）
    date: '2026-06-11'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - docs
    tasks:
      build_cuda_extension: false
      torch_check_validation: false
      document_build_path: false
    artifacts:
      reference: false
      implementation: false
      note: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: rmsnorm_cuda_ext
    phase: implementation
    slots:
      main: '最小 CUDA extension：用 C++/CUDA 重写 rmsnorm（setup.py / TORCH_CHECK / dispatch）'
      depth: 'setup.py vs JIT load 对比 + WSL2 CUDA 12.6 build 路径排查'
      output: 'extension build/run/verify note（含 build 失败时的降级策略）'
  day12:
    title: rmsnorm_cuda_ext 测试 + 与 Triton 版 benchmark
    date: '2026-06-12'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - perf
    tasks:
      add_pytest: false
      add_benchmark: false
      compare_triton_version: false
      add_dtype_dispatch: false
    artifacts:
      tests: false
      benchmark: false
      note: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: rmsnorm_cuda_ext
    phase: benchmark
    slots:
      main: 'extension pytest + benchmark vs Triton 版 rmsnorm（dtype dispatch fp16/bf16/fp32）'
      depth: 'C++ template / dtype dispatch / atomic 写法对比'
      output: 'C++ drill：smart pointer / RAII 复习题 5 道'
  day13:
    title: kv_cache_append toy（→ W5 PagedAttention 铺路）
    date: '2026-06-13'
    status: not_started
    daily_check: 0
    jd_tags:
    - kernel
    - serving
    tasks:
      implement_contiguous_append: false
      add_tests: false
      add_benchmark: false
      paged_layout_design_note: false
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      note: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    operator: kv_cache_append
    phase: benchmark
    slots:
      main: 'kv_cache_append toy（contiguous KV layout）+ tests + bench'
      depth: 'contiguous vs paged 写入差异分析（设计 doc，paged 实际实现留 W5）'
      output: '可选 Triton Tutorial 04 Dropout（PRNG/seeded mask）+ 5 道面经题（CUDA extension / fusion / KV layout）'
  day14:
    title: Week 2 阶段检 + STAR + 全景表
    date: '2026-06-14'
    status: not_started
    daily_check: 0
    weekly_check_score: 0
    stage_check_score: 0
    jd_tags:
    - interview
    tasks:
      operator_mock: false
      cuda_extension_mock: false
      lru_or_queue_drill: false
      cpp_smart_pointer_review: false
      star_metrics_entry: false
      panorama_table: false
    artifacts:
      mock: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
    phase: review
    star_filled: false
    algo_drill_done: false
    cpp_drill_done: false
    slots:
      main: '口头闭卷 mock：fused_add_rmsnorm / swiglu / rope / cuda extension build 流程'
      depth: 'reduction + norm + fusion + extension 全景表（11 个算子的 GB/s / kind / bottleneck / hook）'
      output: 'LRU + smart pointer drill + STAR 段（baseline/metric/improvement/failure/badcase）'
```

- [ ] **Step 2: 验证 week2 解析**

```bash
cd /home/ycy/code/llm-kernel-lab && python -c "import yaml; data=yaml.safe_load(open('study-plan/progress.yaml')); ws=data['week2']; [print(k, '->', v.get('operator','(review)'), '|', v.get('phase')) for k,v in sorted(ws.items())]"
```
Expected: 7 行输出，day08→fused_add_rmsnorm, day09→swiglu, day10→rope, day11→rmsnorm_cuda_ext, day12→rmsnorm_cuda_ext, day13→kv_cache_append, day14→(review)。

- [ ] **Step 3: 跑 driver 三天 smoke test**

```bash
cd /home/ycy/code/llm-kernel-lab && \
python study-plan/run.py day 9 2>&1 | head -20 && echo "---" && \
python study-plan/run.py day 11 2>&1 | head -20 && echo "---" && \
python study-plan/run.py day 14 2>&1 | head -20
```
Expected: 三天分别显示 swiglu / rmsnorm_cuda_ext / (no operator — review/gate day) 与各自的 slots 三行。

- [ ] **Step 4: Commit**

```bash
cd /home/ycy/code/llm-kernel-lab && \
git add study-plan/progress.yaml && \
git commit -m "feat(progress): rewrite week2 days as fusion-extension theme block"
```

---

### Task 5: 替换 `inference-acceleration-plan.md` Day 1-14 表格

**Files:**
- Modify: `study-plan/inference-acceleration-plan.md:60-75`（"56 天安排"表格的 Day 1-14 行）
- Modify: `study-plan/inference-acceleration-plan.md` 末尾（追加 change-log 条目）

- [ ] **Step 1: 打开 `study-plan/inference-acceleration-plan.md`，定位 "## 56 天安排" 表格中 Day 1 到 Day 14 共 14 行（约第 60-73 行），整体替换为下面 14 行**

```markdown
| 1 | 校准 + Nsight Compute WSL2 验证（warmup gate）| ncu 可用性结论 + 算子成熟度审计表 + 重读 Tutorial 01 | `perf`, `docs` |
| 2 | row_softmax baseline + tests + bench（复用 row_max/row_sum 砖块）| reference + impl + tests + bench + Tutorial 02 §1 走读 | `kernel`, `perf` |
| 3 | masked_softmax | impl + tests + bench + warp shuffle vs shared-mem 对比 + Tutorial 02 §2 走读 | `kernel`, `perf` |
| 4 | online_softmax（→ W4 FlashAttention 铺路）| 数学推导 + 单 pass 实现 + tests + bench + ncu profile + 5 道面经题 | `kernel`, `perf` |
| 5 | rmsnorm baseline + tests + bench | reference + impl + tests + bench + vectorized load 对比 + Tutorial 05 forward 走读 | `kernel`, `perf` |
| 6 | layernorm + 与 RMSNorm 对比 | impl + tests + bench + ncu profile + 双对比表 + Tutorial 05 backward 走读 | `kernel`, `perf` |
| 7 | Week 1 周检 + STAR + 家族对比表 | 闭卷 mock + reduction-norm 家族对比表 + TopK/RAII drill + STAR | `interview` |
| 8 | fused_add_rmsnorm（fusion 主题开篇）| residual+norm fusion + tests + bench + ncu + 与 unfused 对比 + vLLM 走读 | `kernel`, `perf` |
| 9 | swiglu + gelu（激活 fusion + Libdevice）| 双 gate + SiLU fusion + tests + bench + Tutorial 07 Libdevice 走读 | `kernel`, `perf` |
| 10 | rope（→ W4 attention 铺路）| RoPE 实现 + tests + bench + sin/cos cache 策略 + vLLM pos_encoding 走读 | `kernel`, `perf` |
| 11 | 最小 PyTorch C++/CUDA extension（rmsnorm 重写）| extension build + setup.py/JIT 对比 + WSL2 build path note | `kernel`, `docs` |
| 12 | rmsnorm_cuda_ext 测试 + 与 Triton 版 benchmark | pytest + bench + dtype dispatch + smart pointer/RAII drill | `kernel`, `perf` |
| 13 | kv_cache_append toy（→ W5 PagedAttention 铺路）| contiguous KV append + tests + bench + paged 设计 doc + 5 道面经题 | `kernel`, `serving` |
| 14 | Week 2 阶段检 + STAR + 全景表 | 闭卷 mock + reduction+norm+fusion+extension 全景表 + LRU/smart pointer drill + STAR | `interview` |
```

- [ ] **Step 2: 在文件末尾"## 相比旧版的主要变更"列表后追加 W1/W2 重设计 change-log 条目**

打开 `study-plan/inference-acceleration-plan.md`，找到"## 相比旧版的主要变更"段落最后一项（应是 14 项）。在该项之后插入：

```markdown
15. **W1/W2 重构为 reduction-normalization-fusion 主题块（2026-05-31）**——算子从 1.5 个扩到 11 个，每天三槽（main/depth/output），对标 Triton Tutorials 02/05/07，hook Day 4 online_softmax → W4、Day 10 rope → W4、Day 13 kv_cache_append → W5。详见 `docs/superpowers/specs/2026-05-31-week1-week2-redesign-design.md`。
```

- [ ] **Step 3: Commit**

```bash
cd /home/ycy/code/llm-kernel-lab && \
git add study-plan/inference-acceleration-plan.md && \
git commit -m "docs(plan): rewrite Day 1-14 table for W1/W2 redesign"
```

---

### Task 6: 端到端 smoke 验证

**Files:**
- 无修改，只跑命令验证整体一致性。

- [ ] **Step 1: 跑全套 study-plan 测试**

```bash
cd /home/ycy/code/llm-kernel-lab && \
python -m pytest tests/test_verify.py tests/test_progress_cli.py tests/test_run_driver.py tests/test_study_plan_dashboard.py tests/test_verify_drill.py -v
```
Expected: 全部 PASS。

- [ ] **Step 2: 用 driver 走一遍 W1/W2 关键日的信息屏**

```bash
cd /home/ycy/code/llm-kernel-lab && \
for d in 1 2 4 7 9 11 13 14; do echo "=== Day $d ==="; python study-plan/run.py day $d 2>&1 | head -20; echo; done
```
Expected：
- Day 1：phase: review, slots 出现"校准"/"重读 Tutorial 01"
- Day 2：operator: row_softmax, slots 出现 tile size sweep
- Day 4：operator: online_softmax, slots 出现"FlashAttention 铺路"
- Day 7：phase: review, slots 出现"家族对比表"
- Day 9：operator: swiglu, slots 出现 Tutorial 07
- Day 11：operator: rmsnorm_cuda_ext, slots 出现 setup.py
- Day 13：operator: kv_cache_append, slots 出现 PagedAttention 铺路
- Day 14：phase: review, slots 出现"全景表"

- [ ] **Step 3: 跑 `today` 命令，确认下一个未启动日合理**

```bash
cd /home/ycy/code/llm-kernel-lab && python study-plan/run.py today 2>&1 | head -20
```
Expected: 输出 `Today: Day 1` 和 Day 1 的 slots（Day 0 是 warmup gate，不算）。

- [ ] **Step 4: 跑 `week 1` 和 `week 2` 总览**

```bash
cd /home/ycy/code/llm-kernel-lab && \
python study-plan/run.py week 1 2>&1 && echo "---" && \
python study-plan/run.py week 2 2>&1
```
Expected: 两周的标题 + operator 列表显示与新表格一致（W1: 5 个 operator + 2 个 review；W2: 6 个 operator + 1 个 review）。

- [ ] **Step 5: 无需 commit（纯验证），结束**

如果上述任一步失败，回到对应 Task 修复并重跑该 Task 的 commit。

---

## Self-Review

**Spec coverage check：**

| Spec 段落 | 落到 Task |
|---|---|
| §"设计原则" 1（算子从 1.5→约 10）| Task 2（9 个新 stub）+ Task 3/4（day operator 字段）|
| §"设计原则" 2（每天三槽）| Task 1（渲染）+ Task 3/4（每天 slots:）|
| §"设计原则" 3（cross-week hook）| Task 3 day04/day10、Task 4 day13 的 title + slots 都点名了 hook |
| §"设计原则" 4（对标 Triton tutorials）| Task 3/4 每个对标日的 output slot 写明 Tutorial 编号 |
| §"设计原则" 5（6 项成熟度）| Task 3/4 每天 artifacts 字段保留原结构，不破坏 verify |
| §"设计原则" 6（Buffer 在周检日）| Task 3 day07、Task 4 day14 phase: review |
| §"Triton 官方 tutorials 集成"表 | Task 3/4 output slot 引用 |
| §"W1 表" 7 天 | Task 3 |
| §"W2 表" 7 天 | Task 4 |
| §"跨周衔接" 4 个 hook | Task 3 day04 / day07 / Task 4 day10 / day13 |
| §"验收标准" 6 项 | 沿用现有 verify schema，无需改动 |
| §"风险与对策" Day 11 降级 | Task 4 day11 slots 的 output 槽点名"含 build 失败时的降级策略"|
| §"不做什么" YAGNI | Task 4 day13 标记 Tutorial 04 为可选；不在 W1/W2 实现 backward |
| §"后续步骤" 1（用户复核 spec）| 已在 brainstorming 阶段完成 |
| §"后续步骤" 2（writing-plans 转实施计划）| 本计划本身 |

**Placeholder scan：** 全文 `grep -E "TBD\|TODO\|fill in\|implement later\|similar to Task"` → 无命中。

**Type/naming consistency：**
- 所有 operator 命名使用 snake_case，与 `verify.resolve_paths` 的 `kernels/triton/<op>/<op>.py` 约定一致。
- `rmsnorm_cuda_ext` 是唯一非 Triton 算子，Task 2 显式给了 `paths:` overrides。
- `slots:` 字段在 Task 1 测试 + Task 3/4 数据 + Task 1 渲染代码三处使用同一结构（`{main, depth, output}`）。
- `phase:` 字段沿用 `reference/tests/benchmark/profile/note/review/implementation` 现有取值。

**Scope check：** 改动局限于 `study-plan/progress.yaml` 的 operators+week1+week2 段、`inference-acceleration-plan.md` Day 1-14 表+change-log、`run.py:_day_show` 末段、fixture 的 day01 块、`test_run_driver.py` 一个新测试。无 verify/dashboard 侧改动，无依赖变更。

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-31-week1-week2-redesign.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — 每个 Task 派一个新 subagent 执行 + 两阶段 review，迭代快、上下文干净。

**2. Inline Execution** — 在当前 session 用 executing-plans 批量执行，带 checkpoint review。

**Which approach?**