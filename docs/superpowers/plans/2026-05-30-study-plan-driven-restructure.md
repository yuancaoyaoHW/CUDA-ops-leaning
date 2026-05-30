# Study-plan-driven 仓库改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `progress.yaml` 升级为带自检的真理源，新增 `study-plan/run.py` daily driver，让 `python study-plan/run.py day N` 成为每天打开仓库的入口。

**Architecture:** 两阶段提交。Phase A：扩 schema（`operators.<op>.kind/paths/commands/thresholds`、`weekN.dayNN.operator/phase`、新增 `week0.day00`、周检日加 `star_filled`/`algo_drill_done`/`cpp_drill_done`），把 `progress.py` 升级成带 `verify` 引擎的工具，扩 dashboard 读侧并锁住 server.py 写入。Phase B：写 `run.py` driver、加 `notes/algorithm-drill.md`、driver 周检 hook 调 `progress.verify` + 解析 STAR/drill markdown。

**Tech Stack:** Python 3.11，PyYAML 6（**不**引入 ruamel.yaml，理由见 Task 0），pytest 9，argparse，stdlib subprocess/glob/re。

**Spec:** `docs/superpowers/specs/2026-05-30-study-plan-driven-restructure-design.md`（commit `63be4df`）。

**重要 yaml 形状约定（实测 progress.yaml）：**
- 不是 `days.dayN`，而是 `weekN.dayNN`（zero-padded 两位）。Day 1 → `week1.day01`；Day 14 → `week2.day14`；Day 56 → `week8.day56`。
- 引入 `week0.day00` 收纳 warmup 算子。
- 顶层不存在 `days:` 键，spec 里的 `days.dayN` 一律读作「`weekN.dayNN`」。

**Spec 偏离声明：**
- spec §3「ruamel.yaml 优先 + PyYAML fallback」改为 **PyYAML only**。理由：当前环境只装了 PyYAML，仓库 `environment.yml` 也只声明了 pyyaml；引入 ruamel 会增加依赖且当前 yaml 没有需要保留的注释。代价：写回时不保 inline 注释——但 `progress.yaml` 是数据文件，没有重要注释。spec 第 2、7、8 节相关描述以本计划为准。

---

## File Structure

**Phase A 创建/修改：**

| 路径 | 责任 | 操作 |
|---|---|---|
| `study-plan/_migration/migrate_yaml.py` | 一次性 schema 迁移脚本 | 新建 |
| `study-plan/progress.yaml` | 升级 schema：operators 加 kind/paths/commands/thresholds；weekN.dayNN 加 operator/phase；新增 week0.day00；周检日加 star_filled/algo_drill_done/cpp_drill_done | 由 migration 脚本改 |
| `study-plan/verify.py` | verify 引擎：路径/命令/门槛 resolver + 6 个 check_* + status 派生 + 回写 | 新建 |
| `study-plan/progress.py` | 新增 `verify`/`status`/`drill` 子命令；`analyze` 切换底层为 verify 输出 | 修改 |
| `study-plan/dashboard.py` | `--build` 调 `verify.verify(write=False)`；新增列渲染 | 修改 |
| `study-plan/server.py` | 拒绝写 artifact / status / 周检 hook 字段 | 修改 |
| `study-plan/dashboard.html` | 渲染 operator/phase 列 + 周检日高亮 | 修改 |
| `tests/test_verify.py` | verify 引擎单元测试 | 新建 |
| `tests/test_progress_cli.py` | progress.py 子命令 smoke test | 新建 |
| `tests/test_study_plan_dashboard.py` | 扩展：新字段渲染 + server.py 写入限制 | 修改 |
| `tests/fixtures/study_plan/` | 迷你 yaml + 假产物用于测试 | 新建 |

**Phase B 创建/修改：**

| 路径 | 责任 | 操作 |
|---|---|---|
| `study-plan/run.py` | daily driver：`day N show/<phase>/done`、`week N show/check`、`today`、`next` | 新建 |
| `notes/algorithm-drill.md` | 周检算法/C++ drill 模板 | 新建 |
| `study-plan/verify.py` | 新增 STAR/drill markdown 解析（`check_star_filled` / `check_drill_done`） | 修改 |
| `study-plan/progress.py` | `drill` 子命令调 STAR/drill 解析 | 修改 |
| `tests/test_run_driver.py` | run.py 子命令单元测试 | 新建 |
| `tests/test_verify_drill.py` | STAR/drill 解析测试 | 新建 |
| `scripts/04_verify_all.sh` | 末尾追加新测试 pytest | 修改 |
| `README.md` | 入口指向 `python study-plan/run.py today` | 修改 |
| `study-plan/README.md` | 新增 driver / verify 一节 | 修改 |
| `AGENTS.md` | "Required Verification" 追加 verify 命令 | 修改 |

**模块依赖：**

```
run.py ──┬── progress.py ──┬── verify.py ──── (subprocess/glob/re)
         └─────────────────┘
dashboard.py ──── verify.py
server.py ──── (不依赖 verify，自己做字段白名单)
```

`verify.py` 是核心；`progress.py` 只做 CLI 包装；`run.py` 只做日级编排。

---

## Phase A: Schema + Verify

### Task 0: 决策记录与依赖确认

**Files:**
- Read: `environment.yml`
- Read: `study-plan/progress.yaml`

- [ ] **Step 1: 确认依赖现状**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -c "import yaml; print(yaml.__version__)"`
Expected: `6.0.3`（已装 PyYAML）

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -c "import ruamel.yaml" 2>&1 | head -1`
Expected: `ModuleNotFoundError: No module named 'ruamel'`

- [ ] **Step 2: 锁定 yaml lib 选择**

确认本计划用 PyYAML（不引入 ruamel）。原因已写在 Plan header 「Spec 偏离声明」。后续所有 `import yaml` 都用 PyYAML。

- [ ] **Step 3: 确认 yaml shape**

Run:
```bash
~/miniconda3/envs/llm-kernel-lab/bin/python -c "
import yaml
d = yaml.safe_load(open('study-plan/progress.yaml'))
print('top:', list(d.keys()))
print('week1 days:', sorted((d.get('week1') or {}).keys())[:3])
print('day01 keys:', list((d['week1']['day01'] or {}).keys()))
"
```
Expected 输出含：
- `top: ['meta', 'operators', 'gpu_libraries', 'week1', ..., 'week8']`
- `week1 days: ['day01', 'day02', 'day03']`
- `day01 keys: ['title', 'date', 'status', 'daily_check', 'jd_tags', 'tasks', 'artifacts', 'verification', 'weaknesses', 'next_fix', 'notes']`

- [ ] **Step 4: 无代码改动，跳过 commit**

本任务只是确认环境与 shape，不动文件。

---

### Task 1: 新建 `tests/fixtures/study_plan/` 测试夹具

**Files:**
- Create: `tests/fixtures/study_plan/mini_progress.yaml`
- Create: `tests/fixtures/study_plan/notes/row_softmax_good.md`
- Create: `tests/fixtures/study_plan/notes/row_softmax_short.md`
- Create: `tests/fixtures/study_plan/reports/json/row_softmax_bench.json`
- Create: `tests/fixtures/study_plan/reports/json/row_softmax_bench_missing_gbps.json`
- Create: `tests/fixtures/study_plan/star_weekly_filled.md`
- Create: `tests/fixtures/study_plan/star_weekly_empty.md`
- Create: `tests/fixtures/study_plan/algo_drill_filled.md`
- Create: `tests/fixtures/study_plan/algo_drill_empty.md`
- Create: `tests/fixtures/study_plan/__init__.py`（空文件，让 pytest discovery 不当 package）

- [ ] **Step 1: 写迷你 progress.yaml**

`tests/fixtures/study_plan/mini_progress.yaml`：

```yaml
meta:
  title: Test Plan
  verify_defaults:
    note_min_lines: 30
    note_required_sections: ["bottleneck", "next_experiment"]
    bench_required_keys_default: ["gbps"]
    bench_required_keys_gemm: ["tflops"]
    profile_extensions: [".ncu-rep", ".nsys-rep"]
    profile_min_size_bytes: 1024
    pytest_timeout_seconds: 60
operators:
  row_softmax:
    kind: reduction
    status: not_started
    artifacts:
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    notes: ''
  axpy:
    kind: elementwise
    status: benchmark_stage
    artifacts:
      reference: true
      implementation: true
      tests: true
      benchmark: true
      profile: false
      note: false
    notes: ''
gpu_libraries:
  triton:
    status: in_progress
    evidence: [row_softmax]
week0:
  day00:
    title: Warmup gate
    operators: [axpy]
    phase: review
    date: ''
    status: not_started
    daily_check: 0
    jd_tags: [kernel, perf]
    tasks: {finish_axpy_note: false}
    artifacts: {}
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
week1:
  day01:
    title: Day One
    operator: row_softmax
    phase: reference
    date: ''
    status: not_started
    daily_check: 0
    jd_tags: [kernel]
    tasks: {audit: false}
    artifacts: {reference: false}
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
  day07:
    title: Week 1 check
    operator: row_softmax
    phase: review
    date: ''
    status: not_started
    daily_check: 0
    weekly_check_score: 0
    star_filled: false
    algo_drill_done: false
    cpp_drill_done: false
    jd_tags: [interview]
    tasks: {weekly_check: false}
    artifacts: {}
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
```

- [ ] **Step 2: 写 note fixtures**

`tests/fixtures/study_plan/notes/row_softmax_good.md`（≥30 行 + 含必填段）：

```markdown
# row_softmax

## Result

baseline: PyTorch row-wise softmax
final: Triton kernel matches reference within 1e-5

## Bottleneck

Memory-bound; GB/s near peak HBM bandwidth.
Block size sweep showed 256 best for hidden=4096.
Larger blocks regress due to register pressure.

## Next experiment

Try fused dropout to amortize the second pass.
Profile at hidden=8192 to confirm peak bandwidth still holds.
Compare against `torch.nn.functional.softmax` with `dim=-1` baseline.

## Correctness

Tested aligned (4096) and non-aligned (4097) shapes.
fp16 / bf16 / fp32 all pass at default tolerance.

## Notes

This kernel uses two-pass online softmax for numerical stability.
The first pass computes max + exp sum; the second writes normalized output.
Block tiling is row-wise; each program handles one row.
Register usage is dominated by per-row accumulators.
Shared memory is unused; reduction stays in registers.
Latency at hidden=4096 is around 12 us on RTX 4060 Laptop.
```

`tests/fixtures/study_plan/notes/row_softmax_short.md`（< 30 行）：

```markdown
# row_softmax

短笔记，只记录 result，没有 bottleneck 段。
```

- [ ] **Step 3: 写 bench JSON fixtures**

`tests/fixtures/study_plan/reports/json/row_softmax_bench.json`：

```json
{
  "operator": "row_softmax",
  "shape": "[1024, 4096]",
  "dtype": "fp16",
  "gbps": 412.5,
  "ms": 0.012
}
```

`tests/fixtures/study_plan/reports/json/row_softmax_bench_missing_gbps.json`：

```json
{
  "operator": "row_softmax",
  "shape": "[1024, 4096]",
  "dtype": "fp16",
  "ms": 0.012
}
```

- [ ] **Step 4: 写 STAR / drill fixtures**

`tests/fixtures/study_plan/star_weekly_filled.md`：

```markdown
# STAR 周记

## Week 1 (2026-06-01 ~ 2026-06-07)

### Situation
本周开了 row_softmax 主线。

### Task
完成 6 项成熟度。

### Action
按 reference / impl / tests / bench / profile / note 顺序推进。

### Result
- baseline: PyTorch
- final: matches within 1e-5
- improvement: latency 18us -> 12us

### Badcase
非对齐 4097 shape 还差 5%。

### Interview-ready
一句话：在 RTX 4060 上把 softmax 写到 ~412 GB/s。

## Week 2 (待填)
```

`tests/fixtures/study_plan/star_weekly_empty.md`：

```markdown
# STAR 周记

## Week 1 (待填)
## Week 2 (待填)
```

`tests/fixtures/study_plan/algo_drill_filled.md`：

```markdown
# 算法 / C++ Drill

## Week 1 (2026-06-01)

### Algo: TopK
- 出处: leetcode 215
- 思路: 用大小为 k 的最小堆维护
- 复杂度: O(n log k)
- 关键代码片段:
  ```cpp
  priority_queue<int, vector<int>, greater<int>> pq;
  ```
- 卡点: heap 初始化用 vector 还是逐个 push 性能差异

### Cpp: RAII
- 主题: RAII 与 unique_ptr
- 学到: 析构函数顺序与异常安全
- 一段最小代码示例:
  ```cpp
  unique_ptr<int> p = make_unique<int>(42);
  ```
- 自测题: 为什么 RAII 比 try/finally 更安全

## Week 2 (待填)
```

`tests/fixtures/study_plan/algo_drill_empty.md`：

```markdown
# 算法 / C++ Drill

## Week 1 (待填)
## Week 2 (待填)
```

- [ ] **Step 5: 写空的 `__init__.py`**

`tests/fixtures/study_plan/__init__.py`：（空文件）

```python
```

- [ ] **Step 6: 验证 fixtures 可被 yaml 解析**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -c "import yaml; print(yaml.safe_load(open('tests/fixtures/study_plan/mini_progress.yaml')).keys())"`
Expected: `dict_keys(['meta', 'operators', 'gpu_libraries', 'week0', 'week1'])`

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/study_plan/
git commit -m "test: add study_plan verify fixtures"
```

---

### Task 2: 创建 `study-plan/verify.py` — 路径/命令/门槛 resolver

**Files:**
- Create: `study-plan/verify.py`
- Create: `tests/test_verify.py`

本任务只实现 resolver 部分（路径推断、门槛合并），不实现 6 个 check_*。check_* 在 Task 3-4 加。

- [ ] **Step 1: 写测试 — 路径默认推断**

`tests/test_verify.py`：

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_verify_module():
    module_path = Path(__file__).resolve().parents[1] / "study-plan" / "verify.py"
    spec = importlib.util.spec_from_file_location("study_plan_verify", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "study_plan"


def test_resolve_paths_uses_conventions_for_kernel():
    verify = load_verify_module()
    op = {"kind": "reduction"}
    paths = verify.resolve_paths("row_softmax", op, repo_root=Path("/repo"))
    assert paths.impl == Path("/repo/kernels/triton/row_softmax/row_softmax.py")
    assert paths.tests == Path("/repo/tests/test_row_softmax.py")
    assert paths.bench == Path("/repo/benchmarks/bench_row_softmax.py")
    assert paths.bench_json == Path("/repo/reports/json/row_softmax_bench.json")
    assert paths.note == Path("/repo/notes/row_softmax.md")
    # profile 是 list of glob patterns（ncu + nsys 任一命中）
    assert any("ncu" in str(p) for p in paths.profile_globs)
    assert any("nsys" in str(p) for p in paths.profile_globs)


def test_resolve_paths_override_wins():
    verify = load_verify_module()
    op = {
        "kind": "reduction",
        "paths": {"impl": "kernels/custom/row_softmax_v2.py"},
    }
    paths = verify.resolve_paths("row_softmax", op, repo_root=Path("/repo"))
    assert paths.impl == Path("/repo/kernels/custom/row_softmax_v2.py")
    # 未 override 的字段仍走默认
    assert paths.tests == Path("/repo/tests/test_row_softmax.py")


def test_resolve_thresholds_merges_defaults_kind_and_override():
    verify = load_verify_module()
    meta = {
        "verify_defaults": {
            "note_min_lines": 30,
            "note_required_sections": ["bottleneck", "next_experiment"],
            "bench_required_keys_default": ["gbps"],
            "bench_required_keys_gemm": ["tflops"],
        }
    }
    op_reduction = {"kind": "reduction"}
    op_gemm = {"kind": "gemm"}
    op_override = {"kind": "reduction", "thresholds": {"note_min_lines": 50}}

    t_red = verify.resolve_thresholds(op_reduction, meta)
    t_gemm = verify.resolve_thresholds(op_gemm, meta)
    t_over = verify.resolve_thresholds(op_override, meta)

    assert t_red.note_min_lines == 30
    assert t_red.bench_required_keys == ["gbps"]
    assert t_gemm.bench_required_keys == ["tflops"]
    assert t_over.note_min_lines == 50
    assert t_red.note_required_sections == ["bottleneck", "next_experiment"]


def test_resolve_command_uses_default_for_kind():
    verify = load_verify_module()
    op = {"kind": "reduction"}
    cmd = verify.resolve_command("row_softmax", op, phase="tests", repo_root=Path("/repo"))
    # default tests 命令: pytest tests/test_<op>.py -v
    assert "pytest" in cmd
    assert "test_row_softmax.py" in cmd


def test_resolve_command_override_wins():
    verify = load_verify_module()
    op = {
        "kind": "reduction",
        "commands": {"tests": "pytest tests/test_row_softmax.py -k aligned -v"},
    }
    cmd = verify.resolve_command("row_softmax", op, phase="tests", repo_root=Path("/repo"))
    assert "-k aligned" in cmd
```

- [ ] **Step 2: 跑测试确认 fail**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify.py -v`
Expected: 5 个测试全 fail（`study-plan/verify.py` 还不存在）。

- [ ] **Step 3: 实现最小 verify.py — resolver 部分**

`study-plan/verify.py`：

```python
"""Verify engine for study-plan progress.yaml.

This module is the source of truth for whether an artifact "really" exists.
progress.py and run.py are thin CLI wrappers on top.

Phase A scope: resolvers + check_* + write_back. STAR/drill checks are added
in Phase B.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---- data classes ----------------------------------------------------------


@dataclass
class Paths:
    impl: Path
    tests: Path
    bench: Path
    bench_json: Path
    note: Path
    profile_globs: list[Path] = field(default_factory=list)


@dataclass
class Thresholds:
    note_min_lines: int
    note_required_sections: list[str]
    bench_required_keys: list[str]
    profile_extensions: list[str]
    profile_min_size_bytes: int
    pytest_timeout_seconds: int


# ---- resolvers -------------------------------------------------------------


def resolve_paths(op_name: str, op: dict[str, Any], repo_root: Path) -> Paths:
    """Resolve artifact paths using convention + optional override."""
    overrides: dict[str, str] = (op.get("paths") or {})

    def pick(key: str, default: Path) -> Path:
        v = overrides.get(key)
        return (repo_root / v) if v else default

    default_impl = repo_root / "kernels" / "triton" / op_name / f"{op_name}.py"
    default_tests = repo_root / "tests" / f"test_{op_name}.py"
    default_bench = repo_root / "benchmarks" / f"bench_{op_name}.py"
    default_bench_json = repo_root / "reports" / "json" / f"{op_name}_bench.json"
    default_note = repo_root / "notes" / f"{op_name}.md"

    profile_globs_override = overrides.get("profile")
    if profile_globs_override:
        profile_globs = [repo_root / profile_globs_override]
    else:
        profile_globs = [
            repo_root / "reports" / "ncu" / f"{op_name}*.ncu-rep",
            repo_root / "reports" / "nsys" / f"bench_{op_name}*.nsys-rep",
        ]

    return Paths(
        impl=pick("impl", default_impl),
        tests=pick("tests", default_tests),
        bench=pick("bench", default_bench),
        bench_json=pick("bench_json", default_bench_json),
        note=pick("note", default_note),
        profile_globs=profile_globs,
    )


def resolve_thresholds(op: dict[str, Any], meta: dict[str, Any]) -> Thresholds:
    """Merge meta.verify_defaults + kind defaults + op.thresholds override."""
    defaults = (meta.get("verify_defaults") or {})
    kind = op.get("kind", "reduction")

    if kind == "gemm":
        bench_keys_default = defaults.get("bench_required_keys_gemm", ["tflops"])
    else:
        bench_keys_default = defaults.get("bench_required_keys_default", ["gbps"])

    override = (op.get("thresholds") or {})
    return Thresholds(
        note_min_lines=override.get("note_min_lines", defaults.get("note_min_lines", 30)),
        note_required_sections=override.get(
            "note_required_sections",
            defaults.get("note_required_sections", ["bottleneck", "next_experiment"]),
        ),
        bench_required_keys=override.get("bench_required_keys", bench_keys_default),
        profile_extensions=defaults.get("profile_extensions", [".ncu-rep", ".nsys-rep"]),
        profile_min_size_bytes=defaults.get("profile_min_size_bytes", 1024),
        pytest_timeout_seconds=defaults.get("pytest_timeout_seconds", 60),
    )


def resolve_command(op_name: str, op: dict[str, Any], phase: str, repo_root: Path) -> str:
    """Resolve shell command for a given (operator, phase)."""
    overrides: dict[str, str] = (op.get("commands") or {})
    if phase in overrides:
        return overrides[phase]

    paths = resolve_paths(op_name, op, repo_root)
    kind = op.get("kind", "reduction")

    # default commands per phase
    if phase == "tests":
        return f"pytest {paths.tests.relative_to(repo_root)} -v"
    if phase == "bench":
        return f"python {paths.bench.relative_to(repo_root)}"
    if phase == "profile":
        if kind in ("attention",):
            return f"bash scripts/run_nsys.sh {op_name}"
        return f"bash scripts/run_ncu.sh {op_name}"
    if phase in ("reference", "impl", "note"):
        # editor-open commands; driver handles them as "open file"
        target = paths.impl if phase != "note" else paths.note
        return f"# open {target.relative_to(repo_root)}"
    raise ValueError(f"unknown phase: {phase}")
```

- [ ] **Step 4: 跑测试确认 pass**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify.py -v`
Expected: 5 个测试全 pass。

- [ ] **Step 5: Commit**

```bash
git add study-plan/verify.py tests/test_verify.py
git commit -m "feat: verify engine resolvers (paths/thresholds/commands)"
```

---

### Task 3: verify.py — 6 个 check_* 函数

**Files:**
- Modify: `study-plan/verify.py`
- Modify: `tests/test_verify.py`

- [ ] **Step 1: 写测试 — check_implementation / check_reference**

在 `tests/test_verify.py` 末尾追加：

```python
def test_check_implementation_file_missing(tmp_path):
    verify = load_verify_module()
    paths = verify.Paths(
        impl=tmp_path / "missing.py",
        tests=tmp_path / "t.py",
        bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json",
        note=tmp_path / "n.md",
        profile_globs=[],
    )
    assert verify.check_implementation(paths, strict=False) is False
    assert verify.check_implementation(paths, strict=True) is False


def test_check_implementation_file_exists_but_empty_strict_fails(tmp_path):
    verify = load_verify_module()
    f = tmp_path / "impl.py"
    f.write_text("")
    paths = verify.Paths(
        impl=f, tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md", profile_globs=[],
    )
    assert verify.check_implementation(paths, strict=False) is True
    assert verify.check_implementation(paths, strict=True) is False


def test_check_implementation_nonempty_passes_both(tmp_path):
    verify = load_verify_module()
    f = tmp_path / "impl.py"
    f.write_text("def foo(): pass\n")
    paths = verify.Paths(
        impl=f, tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md", profile_globs=[],
    )
    assert verify.check_implementation(paths, strict=True) is True


def test_check_reference_strict_requires_torch_or_inline(tmp_path):
    verify = load_verify_module()
    f = tmp_path / "impl.py"
    f.write_text("import triton\n# no torch reference here\n")
    paths = verify.Paths(
        impl=f, tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md", profile_globs=[],
    )
    op = {"kind": "reduction"}
    assert verify.check_reference(op, paths, strict=False) is True  # 文件存在即真
    assert verify.check_reference(op, paths, strict=True) is False

    f.write_text("import torch\nimport triton\n")
    assert verify.check_reference(op, paths, strict=True) is True

    # 显式声明 reference_inline 也算
    f.write_text("import triton\n")
    op2 = {"kind": "reduction", "reference_inline": True}
    assert verify.check_reference(op2, paths, strict=True) is True
```

- [ ] **Step 2: 写测试 — check_benchmark**

继续追加：

```python
def test_check_benchmark_missing_json(tmp_path):
    verify = load_verify_module()
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "missing.json", note=tmp_path / "n.md", profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    assert verify.check_benchmark(paths, th, strict=False) is False
    assert verify.check_benchmark(paths, th, strict=True) is False


def test_check_benchmark_strict_requires_keys(tmp_path):
    verify = load_verify_module()
    bj = tmp_path / "bench.json"
    bj.write_text('{"gbps": 412.0, "ms": 0.012}')
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=bj, note=tmp_path / "n.md", profile_globs=[],
    )
    th_ok = _mk_thresholds(verify, bench_keys=["gbps"])
    th_bad = _mk_thresholds(verify, bench_keys=["tflops"])
    assert verify.check_benchmark(paths, th_ok, strict=True) is True
    assert verify.check_benchmark(paths, th_bad, strict=True) is False
    # non-strict 仅看文件存在
    assert verify.check_benchmark(paths, th_bad, strict=False) is True


def test_check_benchmark_strict_rejects_null_value(tmp_path):
    verify = load_verify_module()
    bj = tmp_path / "bench.json"
    bj.write_text('{"gbps": null}')
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=bj, note=tmp_path / "n.md", profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    assert verify.check_benchmark(paths, th, strict=True) is False


def _mk_thresholds(verify, *, bench_keys):
    return verify.Thresholds(
        note_min_lines=30,
        note_required_sections=["bottleneck", "next_experiment"],
        bench_required_keys=bench_keys,
        profile_extensions=[".ncu-rep", ".nsys-rep"],
        profile_min_size_bytes=1024,
        pytest_timeout_seconds=60,
    )
```

- [ ] **Step 3: 写测试 — check_profile**

继续追加：

```python
def test_check_profile_glob_miss(tmp_path):
    verify = load_verify_module()
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md",
        profile_globs=[tmp_path / "*.ncu-rep"],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    assert verify.check_profile(paths, th, strict=False) is False


def test_check_profile_strict_requires_min_size(tmp_path):
    verify = load_verify_module()
    small = tmp_path / "tiny.ncu-rep"
    small.write_bytes(b"x" * 100)  # <1KB
    big = tmp_path / "real.ncu-rep"
    big.write_bytes(b"x" * 4096)
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md",
        profile_globs=[tmp_path / "*.ncu-rep"],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    # 任意命中 = non-strict pass
    assert verify.check_profile(paths, th, strict=False) is True
    # strict 需要至少一个 >= min_size
    assert verify.check_profile(paths, th, strict=True) is True
    # 把 big 删掉只剩 small，strict 应 fail
    big.unlink()
    assert verify.check_profile(paths, th, strict=True) is False
```

- [ ] **Step 4: 写测试 — check_note**

继续追加：

```python
def test_check_note_missing(tmp_path):
    verify = load_verify_module()
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "missing.md", profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    assert verify.check_note(paths, th, strict=False) is False
    assert verify.check_note(paths, th, strict=True) is False


def test_check_note_short_passes_nonstrict_fails_strict(tmp_path):
    verify = load_verify_module()
    note = tmp_path / "note.md"
    note.write_text("# tiny\n\nshort\n")
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=note, profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    assert verify.check_note(paths, th, strict=False) is True
    assert verify.check_note(paths, th, strict=True) is False


def test_check_note_uses_fixture_good():
    verify = load_verify_module()
    note = FIXTURE_ROOT / "notes" / "row_softmax_good.md"
    paths = verify.Paths(
        impl=Path("/x/i.py"), tests=Path("/x/t.py"), bench=Path("/x/b.py"),
        bench_json=Path("/x/bj.json"), note=note, profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    assert verify.check_note(paths, th, strict=True) is True


def test_check_note_missing_required_section(tmp_path):
    verify = load_verify_module()
    note = tmp_path / "note.md"
    note.write_text("\n".join([f"line {i}" for i in range(35)]) + "\n# bottleneck\nfoo\n")
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=tmp_path / "t.py", bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=note, profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    # 没有 next_experiment 段
    assert verify.check_note(paths, th, strict=True) is False
```

- [ ] **Step 5: 写测试 — check_tests（subprocess fake）**

继续追加：

```python
def test_check_tests_nonstrict_just_file_exists(tmp_path):
    verify = load_verify_module()
    t = tmp_path / "test_x.py"
    t.write_text("def test_ok(): pass\n")
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=t, bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md", profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    assert verify.check_tests("noop", {}, paths, th, strict=False, repo_root=tmp_path) is True


def test_check_tests_strict_runs_pytest_pass(tmp_path):
    verify = load_verify_module()
    t = tmp_path / "test_x.py"
    t.write_text("def test_ok():\n    assert 1 == 1\n")
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=t, bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md", profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    op = {"kind": "reduction"}
    assert verify.check_tests("x", op, paths, th, strict=True, repo_root=tmp_path) is True


def test_check_tests_strict_runs_pytest_fail(tmp_path):
    verify = load_verify_module()
    t = tmp_path / "test_x.py"
    t.write_text("def test_bad():\n    assert 1 == 2\n")
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=t, bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md", profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    op = {"kind": "reduction"}
    assert verify.check_tests("x", op, paths, th, strict=True, repo_root=tmp_path) is False


def test_check_tests_skip_tests_flag(tmp_path):
    verify = load_verify_module()
    t = tmp_path / "test_x.py"
    t.write_text("def test_ok(): assert True\n")
    paths = verify.Paths(
        impl=tmp_path / "i.py", tests=t, bench=tmp_path / "b.py",
        bench_json=tmp_path / "bj.json", note=tmp_path / "n.md", profile_globs=[],
    )
    th = _mk_thresholds(verify, bench_keys=["gbps"])
    op = {"kind": "reduction"}
    # skip_tests=True 时 strict 也直接算 fail（spec §3 strict 表）
    assert verify.check_tests("x", op, paths, th, strict=True, repo_root=tmp_path,
                              skip_tests=True) is False
```

- [ ] **Step 6: 跑测试确认 fail**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify.py -v`
Expected: 新增的 ~14 个测试全 fail（check_* 还没实现）。

- [ ] **Step 7: 实现 6 个 check_***

在 `study-plan/verify.py` 末尾追加：

```python
# ---- check_* ---------------------------------------------------------------


def check_implementation(paths: Paths, *, strict: bool) -> bool:
    if not paths.impl.exists():
        return False
    if strict and paths.impl.stat().st_size == 0:
        return False
    return True


def check_reference(op: dict[str, Any], paths: Paths, *, strict: bool) -> bool:
    if not paths.impl.exists():
        return False
    if not strict:
        return True
    if op.get("reference_inline"):
        return True
    text = paths.impl.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r"\b(import\s+torch|pytorch_reference|torch\.)", text))


def check_tests(
    op_name: str,
    op: dict[str, Any],
    paths: Paths,
    thresholds: Thresholds,
    *,
    strict: bool,
    repo_root: Path,
    skip_tests: bool = False,
) -> bool:
    if not paths.tests.exists():
        return False
    if not strict:
        return True
    if skip_tests:
        return False
    cmd = resolve_command(op_name, op, "tests", repo_root)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(repo_root),
            timeout=thresholds.pytest_timeout_seconds,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        return False
    return proc.returncode == 0


def check_benchmark(paths: Paths, thresholds: Thresholds, *, strict: bool) -> bool:
    if not paths.bench_json.exists():
        return False
    if not strict:
        return True
    try:
        data = json.loads(paths.bench_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    for key in thresholds.bench_required_keys:
        if key not in data:
            return False
        v = data[key]
        if v is None or v == "":
            return False
    return True


def check_profile(paths: Paths, thresholds: Thresholds, *, strict: bool) -> bool:
    matches: list[Path] = []
    for g in paths.profile_globs:
        # g may itself be a glob path (with *); use parent + name pattern
        parent = g.parent
        if not parent.exists():
            continue
        matches.extend(parent.glob(g.name))
    if not matches:
        return False
    if not strict:
        return True
    return any(m.stat().st_size >= thresholds.profile_min_size_bytes for m in matches)


def check_note(paths: Paths, thresholds: Thresholds, *, strict: bool) -> bool:
    if not paths.note.exists():
        return False
    if not strict:
        return True
    text = paths.note.read_text(encoding="utf-8", errors="replace")
    line_count = sum(1 for line in text.splitlines() if line.strip())
    if line_count < thresholds.note_min_lines:
        return False
    lower = text.lower()
    for section in thresholds.note_required_sections:
        if section.lower() not in lower:
            return False
    return True
```

- [ ] **Step 8: 跑测试确认全 pass**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify.py -v`
Expected: 19 个测试全 pass（5 resolver + 14 check_*）。

- [ ] **Step 9: Commit**

```bash
git add study-plan/verify.py tests/test_verify.py
git commit -m "feat: verify engine check_* functions for 6 artifacts"
```

---

### Task 4: verify.py — `verify()` 顶层函数 + status 派生 + 回写

**Files:**
- Modify: `study-plan/verify.py`
- Modify: `tests/test_verify.py`

- [ ] **Step 1: 写测试 — verify 单 operator non-strict 不 write**

在 `tests/test_verify.py` 末尾追加：

```python
import shutil


def _copy_fixture_yaml(tmp_path):
    src = FIXTURE_ROOT / "mini_progress.yaml"
    dst = tmp_path / "progress.yaml"
    shutil.copy(src, dst)
    return dst


def test_verify_single_operator_no_write(tmp_path):
    verify = load_verify_module()
    yaml_path = _copy_fixture_yaml(tmp_path)
    # 不准备任何 artifact 文件 → 全 False
    result = verify.verify(
        yaml_path=yaml_path,
        repo_root=tmp_path,
        target=("operator", "row_softmax"),
        strict=False,
        write=False,
    )
    assert result.operators["row_softmax"].artifacts == {
        "reference": False, "implementation": False, "tests": False,
        "benchmark": False, "profile": False, "note": False,
    }
    # yaml 没动
    import yaml as _y
    raw = _y.safe_load(open(yaml_path))
    assert raw["operators"]["row_softmax"]["status"] == "not_started"


def test_verify_writes_back_artifacts_and_status(tmp_path):
    verify = load_verify_module()
    yaml_path = _copy_fixture_yaml(tmp_path)
    # 准备 row_softmax impl + tests + bench json
    (tmp_path / "kernels" / "triton" / "row_softmax").mkdir(parents=True)
    (tmp_path / "kernels" / "triton" / "row_softmax" / "row_softmax.py").write_text(
        "import torch\nimport triton\n"
    )
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_row_softmax.py").write_text(
        "def test_ok(): assert True\n"
    )
    (tmp_path / "reports" / "json").mkdir(parents=True)
    (tmp_path / "reports" / "json" / "row_softmax_bench.json").write_text(
        '{"gbps": 400}'
    )
    result = verify.verify(
        yaml_path=yaml_path,
        repo_root=tmp_path,
        target=("operator", "row_softmax"),
        strict=False,
        write=True,
    )
    assert result.operators["row_softmax"].artifacts["implementation"] is True
    assert result.operators["row_softmax"].artifacts["benchmark"] is True
    # 回写 yaml
    import yaml as _y
    raw = _y.safe_load(open(yaml_path))
    assert raw["operators"]["row_softmax"]["artifacts"]["implementation"] is True
    assert raw["operators"]["row_softmax"]["artifacts"]["benchmark"] is True
    # status 派生
    assert raw["operators"]["row_softmax"]["status"] in (
        "tests_stage", "benchmark_stage", "impl_stage", "in_progress"
    )


def test_verify_does_not_touch_user_note_fields(tmp_path):
    verify = load_verify_module()
    yaml_path = _copy_fixture_yaml(tmp_path)
    # 写一些用户笔记
    import yaml as _y
    data = _y.safe_load(open(yaml_path))
    data["week1"]["day01"]["weaknesses"] = "kept"
    data["week1"]["day01"]["next_fix"] = "kept-fix"
    data["week1"]["day01"]["daily_check"] = 2
    open(yaml_path, "w").write(_y.safe_dump(data, allow_unicode=True, sort_keys=False))

    verify.verify(
        yaml_path=yaml_path,
        repo_root=tmp_path,
        target=("operator", "row_softmax"),
        strict=False,
        write=True,
    )
    raw = _y.safe_load(open(yaml_path))
    assert raw["week1"]["day01"]["weaknesses"] == "kept"
    assert raw["week1"]["day01"]["next_fix"] == "kept-fix"
    assert raw["week1"]["day01"]["daily_check"] == 2


def test_verify_creates_backup(tmp_path):
    verify = load_verify_module()
    yaml_path = _copy_fixture_yaml(tmp_path)
    verify.verify(
        yaml_path=yaml_path,
        repo_root=tmp_path,
        target=("operator", "row_softmax"),
        strict=False,
        write=True,
    )
    assert (tmp_path / "progress.yaml.bak").exists()


def test_verify_target_all_iterates_all_operators(tmp_path):
    verify = load_verify_module()
    yaml_path = _copy_fixture_yaml(tmp_path)
    result = verify.verify(
        yaml_path=yaml_path,
        repo_root=tmp_path,
        target=("all",),
        strict=False,
        write=False,
    )
    assert set(result.operators.keys()) == {"row_softmax", "axpy"}


def test_derive_status_complete(tmp_path):
    verify = load_verify_module()
    arts = {k: True for k in
            ["reference", "implementation", "tests", "benchmark", "profile", "note"]}
    assert verify.derive_status(arts) == "complete"


def test_derive_status_benchmark_stage(tmp_path):
    verify = load_verify_module()
    arts = {"reference": True, "implementation": True, "tests": True,
            "benchmark": True, "profile": False, "note": False}
    assert verify.derive_status(arts) == "benchmark_stage"


def test_derive_status_not_started(tmp_path):
    verify = load_verify_module()
    arts = {k: False for k in
            ["reference", "implementation", "tests", "benchmark", "profile", "note"]}
    assert verify.derive_status(arts) == "not_started"
```

- [ ] **Step 2: 跑测试确认 fail**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify.py -v -k "verify_ or derive_status"`
Expected: 8 个新测试全 fail。

- [ ] **Step 3: 实现 verify() + derive_status + 回写**

在 `study-plan/verify.py` 末尾追加：

```python
# ---- top-level verify ------------------------------------------------------


@dataclass
class OpResult:
    artifacts: dict[str, bool]
    status: str


@dataclass
class VerifyResult:
    operators: dict[str, OpResult] = field(default_factory=dict)


ARTIFACT_ORDER = ["reference", "implementation", "tests", "benchmark", "profile", "note"]


def derive_status(artifacts: dict[str, bool]) -> str:
    score = sum(1 for k in ARTIFACT_ORDER if artifacts.get(k))
    if score == 6:
        return "complete"
    if score == 5 and not artifacts["note"]:
        return "profile_stage"
    if score == 5 and not artifacts["profile"]:
        return "note_stage"
    if artifacts.get("benchmark"):
        return "benchmark_stage"
    if artifacts.get("tests"):
        return "tests_stage"
    if artifacts.get("implementation"):
        return "impl_stage"
    if score == 0:
        return "not_started"
    return "in_progress"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _backup(yaml_path: Path) -> None:
    bak = yaml_path.with_suffix(yaml_path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(yaml_path.read_bytes())


def _resolve_targets(target: tuple, data: dict[str, Any]) -> list[str]:
    if target[0] == "all":
        return sorted((data.get("operators") or {}).keys())
    if target[0] == "operator":
        return [target[1]]
    if target[0] == "day":
        # day target maps to its operator(s)
        week_key, day_key = _find_day(data, target[1])
        if week_key is None:
            return []
        day = data[week_key][day_key] or {}
        if "operators" in day:
            return list(day["operators"])
        if "operator" in day:
            return [day["operator"]]
        return []
    raise ValueError(f"unknown target: {target}")


def _find_day(data: dict[str, Any], day_num: int) -> tuple[Optional[str], Optional[str]]:
    """Return (week_key, day_key) for given day number, e.g. 1 -> ('week1', 'day01')."""
    day_key = f"day{day_num:02d}"
    for w in range(0, 9):
        wk = f"week{w}"
        if wk in data and day_key in (data.get(wk) or {}):
            return wk, day_key
    return None, None


def verify(
    *,
    yaml_path: Path,
    repo_root: Path,
    target: tuple,
    strict: bool,
    write: bool,
    skip_tests: bool = False,
) -> VerifyResult:
    data = _load_yaml(yaml_path)
    meta = data.get("meta") or {}
    operators = data.get("operators") or {}
    op_names = _resolve_targets(target, data)

    result = VerifyResult()
    for op_name in op_names:
        op = operators.get(op_name)
        if op is None:
            continue
        paths = resolve_paths(op_name, op, repo_root)
        thresholds = resolve_thresholds(op, meta)
        artifacts = {
            "reference": check_reference(op, paths, strict=strict),
            "implementation": check_implementation(paths, strict=strict),
            "tests": check_tests(op_name, op, paths, thresholds,
                                  strict=strict, repo_root=repo_root,
                                  skip_tests=skip_tests),
            "benchmark": check_benchmark(paths, thresholds, strict=strict),
            "profile": check_profile(paths, thresholds, strict=strict),
            "note": check_note(paths, thresholds, strict=strict),
        }
        status = derive_status(artifacts)
        result.operators[op_name] = OpResult(artifacts=artifacts, status=status)

    if write and op_names:
        _backup(yaml_path)
        for op_name, res in result.operators.items():
            data["operators"][op_name]["artifacts"] = res.artifacts
            data["operators"][op_name]["status"] = res.status
        _save_yaml(yaml_path, data)

    return result
```

- [ ] **Step 4: 跑测试确认全 pass**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify.py -v`
Expected: 全部 pass（27 个测试）。

- [ ] **Step 5: Commit**

```bash
git add study-plan/verify.py tests/test_verify.py
git commit -m "feat: verify() top-level + status derivation + yaml writeback"
```

---

### Task 5: Migration 脚本

**Files:**
- Create: `study-plan/_migration/migrate_yaml.py`
- Create: `study-plan/_migration/__init__.py`（空文件）
- Create: `study-plan/_migration/README.md`

**Migration 决策：**
- 只动 schema 形状，不改用户数据。
- `weekN.dayNN` 加 `operator` `phase` 字段（按 `inference-acceleration-plan.md` 表硬编码映射）。
- `operators.<op>` 加 `kind` 字段（reduction/elementwise/gemm/attention/quant 五种）。
- 加 `week0.day00` 块，operators 列 `[vector_add, axpy, row_sum, row_max]`。
- 周检日（day07/14/21/28/35/42/49/56）加 `star_filled: false / algo_drill_done: false / cpp_drill_done: false`。
- `meta` 加 `verify_defaults` 块。

- [ ] **Step 1: 写脚本**

`study-plan/_migration/migrate_yaml.py`：

```python
#!/usr/bin/env python3
"""One-shot migration for study-plan/progress.yaml schema upgrade.

Adds verify-engine fields. Idempotent — safe to re-run; will not overwrite
existing fields.

Usage:
    python study-plan/_migration/migrate_yaml.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRESS_FILE = REPO_ROOT / "study-plan" / "progress.yaml"


VERIFY_DEFAULTS = {
    "note_min_lines": 30,
    "note_required_sections": ["bottleneck", "next_experiment"],
    "bench_required_keys_default": ["gbps"],
    "bench_required_keys_gemm": ["tflops"],
    "profile_extensions": [".ncu-rep", ".nsys-rep"],
    "profile_min_size_bytes": 1024,
    "pytest_timeout_seconds": 60,
}


# operator kinds (manual mapping; review against spec §2)
OPERATOR_KINDS = {
    "vector_add": "elementwise",
    "axpy": "elementwise",
    "row_sum": "reduction",
    "row_max": "reduction",
    "row_softmax": "reduction",
    "rmsnorm": "reduction",
    "flash_attention_toy": "attention",
    "int4_dequant": "quant",
}


# day -> (operator, phase) mapping derived from inference-acceleration-plan.md.
# Phase ∈ {reference, implementation, tests, benchmark, profile, note, review}
DAY_MAP = {
    # week 1: row_softmax
    1: (None, "review"),
    2: ("row_softmax", "reference"),
    3: ("row_softmax", "tests"),
    4: ("row_softmax", "benchmark"),
    5: ("row_softmax", "profile"),
    6: ("row_softmax", "note"),
    7: (None, "review"),
    # week 2: rmsnorm + cuda extension
    8: ("rmsnorm", "reference"),
    9: ("rmsnorm", "implementation"),
    10: ("rmsnorm", "tests"),
    11: ("rmsnorm", "benchmark"),
    12: (None, "implementation"),  # cuda extension demo
    13: (None, "tests"),
    14: (None, "review"),
    # week 3: GEMM (no formal operator in operators block; leave None)
    15: (None, "benchmark"),
    16: (None, "benchmark"),
    17: (None, "profile"),
    18: (None, "review"),
    19: (None, "note"),
    20: (None, "review"),
    21: (None, "review"),
    # week 4: attention
    22: ("flash_attention_toy", "reference"),
    23: ("flash_attention_toy", "reference"),
    24: ("flash_attention_toy", "implementation"),
    25: ("flash_attention_toy", "tests"),
    26: ("flash_attention_toy", "benchmark"),
    27: ("flash_attention_toy", "note"),
    28: (None, "review"),
    # week 5: KV cache / scheduler toy
    29: (None, "implementation"),
    30: (None, "implementation"),
    31: (None, "note"),
    32: (None, "implementation"),
    33: (None, "note"),
    34: (None, "implementation"),
    35: (None, "review"),
    # week 6: vLLM/SGLang/TensorRT-LLM docs
    36: (None, "note"),
    37: (None, "note"),
    38: (None, "note"),
    39: (None, "note"),
    40: (None, "note"),
    41: (None, "note"),
    42: (None, "note"),
    43: (None, "review"),
    # week 7: quant
    44: ("int4_dequant", "reference"),
    45: ("int4_dequant", "implementation"),
    46: ("int4_dequant", "benchmark"),
    47: ("int4_dequant", "implementation"),
    48: ("int4_dequant", "note"),
    49: (None, "review"),
    # week 8: wrap-up
    **{d: (None, "review") for d in range(50, 57)},
}


WEEKLY_CHECK_DAYS = {7, 14, 21, 28, 35, 42, 49, 56}


def migrate(data: dict) -> tuple[dict, list[str]]:
    changes: list[str] = []

    # 1. meta.verify_defaults
    meta = data.setdefault("meta", {})
    if "verify_defaults" not in meta:
        meta["verify_defaults"] = VERIFY_DEFAULTS
        changes.append("added meta.verify_defaults")

    # 2. operators[*].kind
    operators = data.setdefault("operators", {})
    for op_name, op in operators.items():
        if op is None:
            continue
        if "kind" not in op:
            op["kind"] = OPERATOR_KINDS.get(op_name, "reduction")
            changes.append(f"added operators.{op_name}.kind={op['kind']}")

    # 3. week0.day00 — warmup gate
    if "week0" not in data:
        data["week0"] = {
            "day00": {
                "title": "Warmup 算子收尾（Day 1 gate）",
                "operators": ["vector_add", "axpy", "row_sum", "row_max"],
                "phase": "review",
                "date": "",
                "status": "not_started",
                "daily_check": 0,
                "jd_tags": ["kernel", "perf"],
                "tasks": {
                    "finish_axpy_note": False,
                    "finish_row_max_note": False,
                },
                "artifacts": {},
                "verification": "",
                "weaknesses": "",
                "next_fix": "",
                "notes": "",
            }
        }
        # reorder so week0 comes after gpu_libraries and before week1
        data = _reorder_top(data)
        changes.append("added week0.day00 warmup gate")

    # 4. weekN.dayNN: operator/phase + weekly hook fields
    for wnum in range(1, 9):
        wkey = f"week{wnum}"
        week = data.get(wkey) or {}
        for day_key, day in week.items():
            if day is None:
                continue
            day_num = int(day_key.replace("day", ""))
            mapping = DAY_MAP.get(day_num, (None, "review"))
            operator, phase = mapping
            if operator is not None and "operator" not in day:
                day["operator"] = operator
                changes.append(f"{wkey}.{day_key}.operator={operator}")
            if "phase" not in day:
                day["phase"] = phase
                changes.append(f"{wkey}.{day_key}.phase={phase}")
            if day_num in WEEKLY_CHECK_DAYS:
                for fld in ("star_filled", "algo_drill_done", "cpp_drill_done"):
                    if fld not in day:
                        day[fld] = False
                        changes.append(f"{wkey}.{day_key}.{fld}=false")

    return data, changes


def _reorder_top(data: dict) -> dict:
    """Keep top-level key order: meta, operators, gpu_libraries, week0..week8."""
    order = ["meta", "operators", "gpu_libraries"] + [f"week{i}" for i in range(0, 9)]
    new = {}
    for k in order:
        if k in data:
            new[k] = data[k]
    # keep any unknown keys at the end
    for k in data:
        if k not in new:
            new[k] = data[k]
    return new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print changes but don't write yaml")
    parser.add_argument("--file", type=Path, default=PROGRESS_FILE,
                        help="progress.yaml to migrate")
    args = parser.parse_args()

    data = yaml.safe_load(args.file.read_text(encoding="utf-8"))
    new_data, changes = migrate(data)

    if not changes:
        print("no changes (already migrated)")
        return 0

    for c in changes:
        print(f"  + {c}")

    if args.dry_run:
        print(f"\n[dry-run] would write {len(changes)} changes")
        return 0

    bak = args.file.with_suffix(args.file.suffix + ".pre-migration.bak")
    bak.write_bytes(args.file.read_bytes())
    args.file.write_text(
        yaml.safe_dump(new_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nwrote {args.file} (backup: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 写 README**

`study-plan/_migration/README.md`：

```markdown
# Migration scripts

One-shot scripts that mutate `progress.yaml` schema. Run once, then ignore.

## migrate_yaml.py

Phase A schema upgrade (2026-05-30 spec):

- adds `meta.verify_defaults`
- adds `operators.<op>.kind` (manual mapping)
- adds `week0.day00` warmup gate
- adds `weekN.dayNN.operator/phase`
- adds `star_filled / algo_drill_done / cpp_drill_done` on weekly check days

Idempotent. Run with `--dry-run` first.
```

- [ ] **Step 3: 写空 `__init__.py`**

`study-plan/_migration/__init__.py`：（空文件）

```python
```

- [ ] **Step 4: Dry-run**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/_migration/migrate_yaml.py --dry-run`
Expected 输出形如：
```
  + added meta.verify_defaults
  + added operators.vector_add.kind=elementwise
  + added operators.axpy.kind=elementwise
  ... (~70 行 changes)
[dry-run] would write 70 changes
```

- [ ] **Step 5: 真跑迁移**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/_migration/migrate_yaml.py`
Expected: 末尾 `wrote .../progress.yaml (backup: progress.yaml.pre-migration.bak)`

- [ ] **Step 6: 校验迁移结果**

Run:
```bash
~/miniconda3/envs/llm-kernel-lab/bin/python -c "
import yaml
d = yaml.safe_load(open('study-plan/progress.yaml'))
assert 'verify_defaults' in d['meta']
assert d['operators']['row_softmax']['kind'] == 'reduction'
assert 'week0' in d
assert d['week0']['day00']['operators'] == ['vector_add', 'axpy', 'row_sum', 'row_max']
assert d['week1']['day02']['operator'] == 'row_softmax'
assert d['week1']['day02']['phase'] == 'reference'
assert d['week1']['day07']['star_filled'] is False
print('migration ok')
"
```
Expected: `migration ok`

- [ ] **Step 7: 再 dry-run 验证幂等**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/_migration/migrate_yaml.py --dry-run`
Expected: `no changes (already migrated)`

- [ ] **Step 8: Commit**

```bash
git add study-plan/_migration/ study-plan/progress.yaml
git commit -m "feat: migrate progress.yaml schema for verify engine"
```

注意：不 commit `progress.yaml.pre-migration.bak`，它是本地备份。如果 `.gitignore` 没拦住，加一行 `study-plan/*.bak` 到 `.gitignore` 一并提交。

---

### Task 6: `progress.py` 接 verify 引擎（新子命令 + analyze 切底层）

**Files:**
- Modify: `study-plan/progress.py`
- Create: `tests/test_progress_cli.py`

`progress.py` 仍是 CLI 入口，但底层的 artifact 状态全部走 `verify.verify()`。`update`/`week`/`history` 子命令保持不动。

- [ ] **Step 1: 写失败测试 — verify 子命令存在**

`tests/test_progress_cli.py`：

```python
from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "study_plan"


def run_progress(args, cwd):
    return subprocess.run(
        [sys.executable, str(REPO / "study-plan" / "progress.py")] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def setup_workspace(tmp_path):
    """Copy fixture into tmp_path/study-plan/progress.yaml."""
    sp = tmp_path / "study-plan"
    sp.mkdir()
    shutil.copy(FIXTURES / "mini_progress.yaml", sp / "progress.yaml")
    (tmp_path / "kernels" / "triton" / "row_softmax").mkdir(parents=True)
    (tmp_path / "kernels" / "triton" / "row_softmax" / "row_softmax.py").write_text(
        "import torch\n\ndef pytorch_reference(x): return torch.softmax(x, -1)\n"
    )
    return sp


def test_verify_cli_default_all(tmp_path, monkeypatch):
    setup_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(REPO / "study-plan"))
    result = run_progress(["verify"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "row_softmax" in result.stdout


def test_verify_cli_operator(tmp_path, monkeypatch):
    setup_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(REPO / "study-plan"))
    result = run_progress(["verify", "--operator", "row_softmax"], cwd=tmp_path)
    assert result.returncode == 0
    assert "row_softmax" in result.stdout


def test_verify_cli_unknown_operator_exits_2(tmp_path, monkeypatch):
    setup_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(REPO / "study-plan"))
    result = run_progress(["verify", "--operator", "no_such_op"], cwd=tmp_path)
    assert result.returncode == 2
    assert "no_such_op" in result.stderr or "no_such_op" in result.stdout


def test_status_cli_day(tmp_path, monkeypatch):
    setup_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(REPO / "study-plan"))
    result = run_progress(["status", "--day", "2"], cwd=tmp_path)
    assert result.returncode == 0
    assert "day02" in result.stdout or "Day 2" in result.stdout


def test_legacy_subcommands_still_work(tmp_path, monkeypatch):
    setup_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(REPO / "study-plan"))
    # `analyze` must still print sections (现在底层走 verify)
    result = run_progress(["analyze"], cwd=tmp_path)
    assert result.returncode == 0
    assert "JD" in result.stdout or "标签" in result.stdout
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_progress_cli.py -v`
Expected: 全部 FAIL（`verify` / `status` 子命令还没加；`analyze` 跑得过但行为可能不变 — 这一步重点是新增子命令）

- [ ] **Step 3: 在 `progress.py` 顶部加 argparse 入口（替换现有 main）**

Edit `study-plan/progress.py`，把 `main()` 整体替换为下面这版。其它函数（`load_progress`、`show_dashboard`、`show_week_detail`、`show_history`、`show_analysis`、`interactive_update`）保持不动。

```python
import argparse


def cmd_verify(args, data):
    import verify  # study-plan/verify.py
    if args.operator:
        targets = ("operator", args.operator)
    elif args.day is not None:
        targets = ("day", args.day)
    else:
        targets = ("all", None)
    try:
        results = verify.verify(
            data,
            target=targets,
            strict=args.strict,
            write=args.write,
            skip_tests=args.skip_tests,
            repo_root=Path(__file__).resolve().parents[1],
        )
    except verify.UnknownTargetError as exc:
        print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
        sys.exit(2)
    except verify.MissingDependencyError as exc:
        print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
        sys.exit(4)

    if args.write:
        try:
            verify.write_back(data, results, PROGRESS_FILE)
        except verify.BackupError as exc:
            print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
            sys.exit(3)

    verify.print_summary(results, strict=args.strict)
    return 0


def cmd_status(args, data):
    import verify
    if args.day is None:
        print(f"{C.RED}--day N is required{C.RESET}", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = verify.verify_day(data, args.day, strict=False, skip_tests=True, repo_root=repo_root)
    except verify.UnknownTargetError as exc:
        print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
        return 2
    verify.print_day_status(result, args.day)
    return 0


def cmd_drill(args, data):
    import verify
    repo_root = Path(__file__).resolve().parents[1]
    summary = verify.collect_drill_summary(data, repo_root=repo_root)
    verify.print_drill_summary(summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="progress")
    sub = parser.add_subparsers(dest="cmd")

    p_verify = sub.add_parser("verify", help="Run verify engine")
    target = p_verify.add_mutually_exclusive_group()
    target.add_argument("--day", type=int)
    target.add_argument("--operator", type=str)
    target.add_argument("--all", action="store_true")
    p_verify.add_argument("--strict", action="store_true")
    p_verify.add_argument("--write", action="store_true")
    p_verify.add_argument("--skip-tests", action="store_true")

    p_status = sub.add_parser("status", help="Show single-day artifact status")
    p_status.add_argument("--day", type=int)

    sub.add_parser("drill", help="Show STAR/algo/cpp drill summary")

    sub.add_parser("week", help="Show current week detail")
    sub.add_parser("history", help="Show recent history")
    sub.add_parser("analyze", help="Show coverage analysis")
    sub.add_parser("update", help="Interactive update (legacy)")

    args = parser.parse_args()
    data = load_progress()

    if args.cmd == "verify":
        return cmd_verify(args, data)
    if args.cmd == "status":
        return cmd_status(args, data)
    if args.cmd == "drill":
        return cmd_drill(args, data)
    if args.cmd == "week":
        show_week_detail(data); return 0
    if args.cmd == "history":
        show_history(data); return 0
    if args.cmd == "analyze":
        show_analysis(data); return 0
    if args.cmd == "update":
        interactive_update(data); return 0

    show_dashboard(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 让 `show_analysis` 走 verify 输出**

找到 `progress.py` 里 `show_analysis(data)` 函数（按 `grep -n "def show_analysis" study-plan/progress.py` 定位）。在函数开头插入：

```python
def show_analysis(data: dict[str, Any]) -> None:
    # 切到 verify 真值再统计：旧逻辑读 yaml 字段，新逻辑先 verify 再读
    import verify
    repo_root = Path(__file__).resolve().parents[1]
    try:
        results = verify.verify(
            data,
            target=("all", None),
            strict=False,
            write=False,
            skip_tests=True,
            repo_root=repo_root,
        )
        verify.apply_results_in_memory(data, results)
    except Exception as exc:
        print(f"{C.YELLOW}verify 失败，使用 yaml 静态字段：{exc}{C.RESET}")
    # ... (原有渲染逻辑保持不动)
```

`verify.apply_results_in_memory` 在 Task 4 里已实现：把 verify 结果写到 in-memory dict 的 `artifacts.*`、`status` 字段，但**不写盘**。这样 `show_analysis` 的下游统计（JD 标签、算子成熟度、GPU 库覆盖度）都用真值。

- [ ] **Step 5: 跑测试确认通过**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_progress_cli.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: 跑回归测试**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify.py tests/test_progress_cli.py -v`
Expected: 全部 PASS。

- [ ] **Step 7: 手动 smoke**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/progress.py verify`
Expected: 列出 8 个算子的 0-6/6 状态，row_sum 满 6（已有真证据），row_softmax 0（未起步）。退出码 0。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/progress.py analyze`
Expected: 输出和迁移前类似（标签覆盖度、算子成熟度），但底层是 verify 真值。

- [ ] **Step 8: Commit**

```bash
git add study-plan/progress.py tests/test_progress_cli.py
git commit -m "feat: progress.py CLI subcommands verify/status/drill"
```

---

### Task 7: Dashboard 读侧适配新 schema

**Files:**
- Modify: `study-plan/dashboard.py`
- Modify: `study-plan/dashboard.html`
- Modify: `tests/test_study_plan_dashboard.py`

dashboard 在 `--build` 时调 `verify.verify(write=False)`，让模板拿到「verify 真值」而不是 yaml 静态字段。新增 operator、phase、star_filled、algo_drill_done、cpp_drill_done 列。

- [ ] **Step 1: 写失败测试 — 新字段在渲染输出里**

Append to `tests/test_study_plan_dashboard.py`：

```python
def test_dashboard_build_includes_operator_and_phase(tmp_path, monkeypatch):
    """--build 应当把每天的 operator/phase 字段渲染到 HTML。"""
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    # 用 mini_progress fixture
    fixture = Path(__file__).resolve().parent / "fixtures" / "study_plan" / "mini_progress.yaml"
    progress_file.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    out = tmp_path / "dashboard_out.html"
    monkeypatch.setattr(dashboard, "OUTPUT_FILE", out)

    dashboard.build_static_dashboard()
    html = out.read_text(encoding="utf-8")
    assert "row_softmax" in html
    assert "reference" in html  # phase 名


def test_dashboard_build_uses_verify_truth(tmp_path, monkeypatch):
    """fixture 里 row_softmax artifacts 全部 false；
    但 mini_progress + 真存在 reference 文件后，build 渲染应当显示 reference=true。"""
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    fixture = Path(__file__).resolve().parent / "fixtures" / "study_plan" / "mini_progress.yaml"
    progress_file.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    # 在 tmp_path 下造一个 row_softmax impl 文件，让 verify reference=True
    (tmp_path / "kernels" / "triton" / "row_softmax").mkdir(parents=True)
    (tmp_path / "kernels" / "triton" / "row_softmax" / "row_softmax.py").write_text(
        "import torch\n\ndef pytorch_reference(x): return torch.softmax(x, -1)\n"
    )
    monkeypatch.chdir(tmp_path)

    out = tmp_path / "dashboard_out.html"
    monkeypatch.setattr(dashboard, "OUTPUT_FILE", out)

    dashboard.build_static_dashboard()
    html = out.read_text(encoding="utf-8")
    # 期望 reference 渲染成 ✓ 而不是 ✗（具体 token 看模板，下面任选其一）
    assert "ref-true" in html or 'reference: true' in html or '✓ reference' in html
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_study_plan_dashboard.py -v -k "operator or verify_truth"`
Expected: FAIL（dashboard 还没切走 verify，也没渲染 operator/phase）。

- [ ] **Step 3: 在 `dashboard.py` 切底层**

定位 `dashboard.py` 里读 progress 的入口（一般是 `build_static_dashboard()` 或类似函数）。在加载 yaml 之后、渲染之前，加一段：

```python
def _apply_verify(data):
    """dashboard 渲染前先用 verify 把 artifacts 状态算成真值。"""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent))
        import verify  # study-plan/verify.py
        repo_root = _Path(__file__).resolve().parents[1]
        results = verify.verify(
            data,
            target=("all", None),
            strict=False,
            write=False,
            skip_tests=True,
            repo_root=repo_root,
        )
        verify.apply_results_in_memory(data, results)
    except Exception as exc:
        # 不让 dashboard 因为 verify 报错就挂掉
        print(f"[dashboard] verify failed, falling back to yaml static: {exc}")
    return data
```

然后在 `build_static_dashboard()` 里 `data = load_progress()` 之后立刻 `data = _apply_verify(data)`。

- [ ] **Step 4: 把 operator/phase 注入 day 渲染**

在 `dashboard.py` 里渲染每日 card 的位置（grep `daily_check` 或 `tasks` 周围），增加：

```python
def render_day(day_key, day_data):
    op = day_data.get("operator", "")
    phase = day_data.get("phase", "")
    is_weekly = day_key in {"day07", "day14", "day21", "day28", "day35", "day42", "day49", "day56"}
    star = day_data.get("star_filled", False) if is_weekly else None
    algo = day_data.get("algo_drill_done", False) if is_weekly else None
    cpp = day_data.get("cpp_drill_done", False) if is_weekly else None
    # 把 op/phase/star/algo/cpp 织入现有 HTML 模板字符串
    # 关键最小渲染：在 card title 下加一行
    #   <div class="day-meta">op={op} · phase={phase}{weekly_chips}</div>
    ...
```

具体 HTML 字符串怎么织取决于现有 `dashboard.py` 模板写法。最小改动原则：找 card 的 `<h3>` 或 `<div class="day-...">`，在后面 append 一段 `<div class="day-meta">op=... · phase=...</div>`。周检日 chip 用三个 `<span class="chip {star/algo/cpp}-{ok|miss}">`。

- [ ] **Step 5: 加 reference token 让测试 assertion 命中**

确保 `render_day` 或 artifact bar 输出形如：

```html
<span class="ref-{state}">reference</span>
```

其中 state ∈ {true, false}，对应 `day_data.get("artifacts", {}).get("reference")` 或 operator artifacts。这样 `test_dashboard_build_uses_verify_truth` 的 `'ref-true' in html` 才会命中。如果 dashboard 已有 ✓/✗ 渲染，可以保留 ✓ 并把 assertion token 调成 `'✓ reference'`，但保持一致。

- [ ] **Step 6: 跑测试确认通过**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_study_plan_dashboard.py -v`
Expected: 新加的两个测试 PASS，旧测试也 PASS。

- [ ] **Step 7: 手动 smoke**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/dashboard.py --build`
Expected: 生成 `study-plan/index.html`（或既有输出文件），打开浏览器看 day cards 多了 `op=… · phase=…` 一行；周检日卡片有 STAR/algo/cpp chip。

- [ ] **Step 8: Commit**

```bash
git add study-plan/dashboard.py study-plan/dashboard.html tests/test_study_plan_dashboard.py
git commit -m "feat: dashboard reads verify truth + renders operator/phase/drill chips"
```

注：`dashboard.html` 如果不是模板而是输出产物，则不 commit；只 commit `dashboard.py`。Task 9 会确认 `dashboard.html` 在仓库里的角色。

---

### Task 8: `server.py` 锁住 artifact 字段写入

**Files:**
- Modify: `study-plan/server.py`
- Modify: `tests/test_study_plan_dashboard.py`

`server.py` 是 dashboard 的编辑后端。spec 决定：artifact / status / 周检 hook 字段一律不允许通过 web 编辑——必须靠 `run.py done` + verify 写回。

- [ ] **Step 1: 读 server.py 当前形态**

Run: `cat study-plan/server.py`
确认它怎么接收 PUT/POST 请求、怎么 dispatch 到 `dashboard.update_day` 之类函数。下面假设它把请求 JSON 直接传给 `update_day`。

- [ ] **Step 2: 写失败测试**

Append to `tests/test_study_plan_dashboard.py`：

```python
def test_update_day_rejects_artifact_fields(tmp_path, monkeypatch):
    """artifact / status / star_filled 等字段必须被 update_day 拒绝（或忽略）。"""
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    payload = {
        "daily_check": 2,                    # 允许
        "weaknesses": "test",                # 允许
        "artifacts": {"docs": True},         # 拒绝
        "status": "complete",                # 拒绝
        "star_filled": True,                 # 拒绝
    }

    ok = dashboard.update_day(1, payload)
    assert ok is True

    saved = yaml.safe_load(progress_file.read_text(encoding="utf-8"))
    day = saved["week1"]["day01"]
    assert day["daily_check"] == 2
    assert day["weaknesses"] == "test"
    # 这些字段不允许被 server 改
    assert day["status"] == "not_started"
    assert day["artifacts"] == {"docs": False}


def test_update_operator_artifact_rejected(tmp_path, monkeypatch):
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    # 假设 server 也暴露 update_operator；如果没有这个函数就跳过
    if not hasattr(dashboard, "update_operator"):
        return
    ok = dashboard.update_operator(
        "row_softmax",
        {"status": "complete", "artifacts": {"reference": True}, "notes": "ok"},
    )
    assert ok is True
    saved = yaml.safe_load(progress_file.read_text(encoding="utf-8"))
    op = saved["operators"]["row_softmax"]
    assert op["notes"] == "ok"
    assert op["status"] == "not_started"
    assert op["artifacts"]["reference"] is False
```

- [ ] **Step 3: 跑测试确认失败**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_study_plan_dashboard.py -v -k "rejects or rejected"`
Expected: FAIL。

- [ ] **Step 4: 在 `dashboard.py` 加字段白名单**

定位 `dashboard.py` 里的 `update_day` 函数。在函数开头加：

```python
USER_NOTE_FIELDS_DAY = {
    "daily_check",
    "date",
    "verification",
    "weaknesses",
    "next_fix",
    "notes",
    "tasks",                  # 用户勾选 tasks 仍允许
    "weekly_check_score",
    "stage_check_score",
}
USER_NOTE_FIELDS_OPERATOR = {"notes"}


def _filter_user_fields(payload, allowed):
    return {k: v for k, v in payload.items() if k in allowed}


def update_day(day_number, payload):
    payload = _filter_user_fields(payload, USER_NOTE_FIELDS_DAY)
    # ... 原有逻辑（status 派生那段也要去掉，或改成只在所有字段允许时跑；
    # 简化：strip status 派生，因为 status 现在由 verify 决定）
```

如果 `dashboard.update_day` 现有逻辑里有「全部 task/artifact 完成则 status=done」，**整段删掉**——status 现在只由 verify 写。

类似地，如果有 `update_operator` 就加白名单 `USER_NOTE_FIELDS_OPERATOR`；没有就不加。

- [ ] **Step 5: 跑测试确认通过**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_study_plan_dashboard.py -v`
Expected: 全 PASS。

- [ ] **Step 6: 手动 smoke web**

Run（后台）: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/server.py &`
然后 `curl -X POST http://localhost:PORT/day/1 -d '{"status":"done","artifacts":{"docs":true}}' -H 'content-type: application/json'`（PORT 看 server.py 实际值）。
Expected: HTTP 200，但 yaml 里 status 仍是 `not_started`，artifacts.docs 仍是 false。
（如果 server.py 是简单 http.server，可能没 POST endpoint。这一步不强制——单元测试已覆盖。）

Kill server: `pkill -f study-plan/server.py`

- [ ] **Step 7: Commit**

```bash
git add study-plan/dashboard.py tests/test_study_plan_dashboard.py
git commit -m "feat: server only accepts user-note fields; verify owns artifact/status"
```

---

### Task 9: Phase A 收尾 — 全量回归 + 手动验证

**Files:**
- Run only

- [ ] **Step 1: 全测试**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/ -v`
Expected: 全 PASS。

- [ ] **Step 2: verify 真仓库**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/progress.py verify --operator row_sum --strict`
Expected: 6/6（已有 ncu profile 和 note）。退出码 0。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/progress.py verify --operator axpy --strict`
Expected: ≤5/6。明确指出 note partial（行数不足或缺段）或 profile 缺。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/progress.py verify --operator row_softmax`
Expected: 0/6（未起步）。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/progress.py verify --all --write`
Run: `git diff study-plan/progress.yaml`
Expected: diff 只动 `operators.<op>.artifacts.*`、`operators.<op>.status`、`week*.day*.artifacts.*`、`week*.day*.status`。**不**动 `verification` / `weaknesses` / `next_fix` / `notes` / `daily_check`。

如果 diff 里有用户笔记字段被改，回滚 yaml 并修 verify.py write_back 白名单。

- [ ] **Step 3: dashboard 渲染**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/dashboard.py --build`
打开 `study-plan/index.html`，目测：
- Day cards 多了 `op=… · phase=…`
- 周检日（day07/14/21/...）有 STAR/algo/cpp chip（虽然全是 miss）
- row_sum 算子卡片显示 6/6

- [ ] **Step 4: 决定是否保留 verify --write 的副作用**

如果 Step 2 的 `--write` 改了真 progress.yaml，决定提交还是回滚：
- 提交：`git commit -am "chore: verify --write sync progress.yaml truth"`
- 回滚：`git checkout study-plan/progress.yaml`

推荐回滚——Phase A 的目标是引擎能用，不是先把状态推到最新；状态更新留到日常使用 `run.py done` 触发。

- [ ] **Step 5: Phase A 完工 commit（如果还有 staged 改动）**

Run: `git status -s`
如有未提交（比如 dashboard.html 输出），决定是否要 commit。原则：
- 工具代码、测试、配置 commit
- 输出产物（rendered html）按现有仓库习惯（如果之前 commit 过就 commit）

至此 Phase A 结束。引擎可用，dashboard 显示真值，server.py 不再写 artifact 字段。

---

## Phase B: Driver + 周检 Hooks

### Task 10: 新建 `notes/algorithm-drill.md` 模板 + STAR/drill 解析

**Files:**
- Create: `notes/algorithm-drill.md`
- Modify: `study-plan/verify.py`（加 `check_star_filled` / `check_drill_done` / `collect_drill_summary`）
- Create: `tests/test_verify_drill.py`

- [ ] **Step 1: 写 `notes/algorithm-drill.md`**

```markdown
# 算法 / C++ Drill 周记

每周周检（Day 7/14/21/28/35/42/49/56）做 1 道算法 + 1 道 C++/系统基础。
60-90 分钟内完成；不抢主线。drill 表见 study-plan/inference-acceleration-plan.md。

模板：

## Week N (YYYY-MM-DD)

### Algo: <题名>
- 题目链接 / 出处:
- 思路（一句话）:
- 复杂度: O(...)
- 关键代码片段:
  ```cpp
  // ...
  ```
- 卡点 / 复盘:

### Cpp: <主题>
- 主题（如 RAII / smart pointer）:
- 学到的 1 个反直觉点:
- 一段最小代码示例:
  ```cpp
  // ...
  ```
- 自测题（口述）:

---

## Week 1 (待填)

## Week 2 (待填)

## Week 3 (待填)

## Week 4 (待填)

## Week 5 (待填)

## Week 6 (待填)

## Week 7 (待填)

## Week 8 (待填)
```

- [ ] **Step 2: 写失败测试**

`tests/test_verify_drill.py`：

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "study_plan"


def load_verify():
    spec = importlib.util.spec_from_file_location("verify", REPO / "study-plan" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_star_filled_true(tmp_path):
    verify = load_verify()
    star = tmp_path / "star.md"
    star.write_text((FIXTURES / "star_weekly_filled.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert verify.check_star_filled(star, week=1, strict=False) is True


def test_check_star_filled_false_when_only_template(tmp_path):
    verify = load_verify()
    star = tmp_path / "star.md"
    star.write_text((FIXTURES / "star_weekly_empty.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert verify.check_star_filled(star, week=1, strict=False) is False


def test_check_star_strict_requires_subsections(tmp_path):
    verify = load_verify()
    star = tmp_path / "star.md"
    # 用 _filled fixture：里面应该已经包含 Situation/Task/Action/Result/Badcase
    star.write_text((FIXTURES / "star_weekly_filled.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert verify.check_star_filled(star, week=1, strict=True) is True


def test_check_drill_done_algo_and_cpp(tmp_path):
    verify = load_verify()
    drill = tmp_path / "drill.md"
    drill.write_text((FIXTURES / "algo_drill_filled.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert verify.check_drill_done(drill, week=1, kind="algo") is True
    assert verify.check_drill_done(drill, week=1, kind="cpp") is True


def test_check_drill_done_empty(tmp_path):
    verify = load_verify()
    drill = tmp_path / "drill.md"
    drill.write_text((FIXTURES / "algo_drill_empty.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert verify.check_drill_done(drill, week=1, kind="algo") is False
    assert verify.check_drill_done(drill, week=1, kind="cpp") is False
```

- [ ] **Step 3: 跑测试确认失败**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify_drill.py -v`
Expected: FAIL（函数未定义）。

- [ ] **Step 4: 在 `verify.py` 加解析函数**

```python
import re

WEEK_HEADING_RE = re.compile(r"^## Week (\d+)\b", re.MULTILINE)
SUB_HEADING_RE = re.compile(r"^### (Algo|Cpp|Situation|Task|Action|Result|Badcase)\b", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\(\s*待填\s*\)")
STAR_REQUIRED_SUBSECTIONS = ("Situation", "Task", "Action", "Result", "Badcase")


def _extract_week_section(md_text: str, week: int) -> str | None:
    """返回 ## Week N 段（不含标题行）到下一个 ## Week 之间的文本，找不到返回 None."""
    matches = list(WEEK_HEADING_RE.finditer(md_text))
    for i, m in enumerate(matches):
        if int(m.group(1)) != week:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        return md_text[start:end]
    return None


def _significant_lines(text: str) -> list[str]:
    """去掉空行和 (待填) 占位的非空行。"""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if PLACEHOLDER_RE.search(s) and len(s) <= 15:  # 标题级"(待填)"
            continue
        out.append(s)
    return out


def check_star_filled(path: Path, week: int, *, strict: bool) -> bool:
    if not path.exists():
        return False
    section = _extract_week_section(path.read_text(encoding="utf-8"), week)
    if section is None:
        return False
    sig = _significant_lines(section)
    if len(sig) < 5:
        return False
    if strict:
        for sub in STAR_REQUIRED_SUBSECTIONS:
            if not re.search(rf"^### {sub}\b", section, re.MULTILINE):
                return False
    return True


def check_drill_done(path: Path, week: int, *, kind: str) -> bool:
    """kind ∈ {'algo', 'cpp'}."""
    if not path.exists():
        return False
    section = _extract_week_section(path.read_text(encoding="utf-8"), week)
    if section is None:
        return False
    sub_label = "Algo" if kind == "algo" else "Cpp"
    sub_re = re.compile(rf"^### {sub_label}\b.*?(?=^### |\Z)", re.MULTILINE | re.DOTALL)
    sub_match = sub_re.search(section)
    if not sub_match:
        return False
    sub_text = sub_match.group(0)
    sig = _significant_lines(sub_text)
    # ≥ 3 非空行（标题行算 1 行，至少要 2 行实际内容）
    return len(sig) >= 3


def collect_drill_summary(data: dict, *, repo_root: Path) -> dict:
    """返回 {week: {star: bool, algo: bool, cpp: bool}}."""
    star_path = repo_root / "notes" / "star-weekly.md"
    drill_path = repo_root / "notes" / "algorithm-drill.md"
    out = {}
    for w in range(1, 9):
        out[w] = {
            "star": check_star_filled(star_path, w, strict=False),
            "algo": check_drill_done(drill_path, w, kind="algo"),
            "cpp": check_drill_done(drill_path, w, kind="cpp"),
        }
    return out


def print_drill_summary(summary: dict) -> None:
    print("Week | STAR | Algo | Cpp")
    print("-----|------|------|----")
    for w, row in summary.items():
        marks = lambda b: "  ✓" if b else "  ✗"
        print(f"  {w}  | {marks(row['star'])} | {marks(row['algo'])} | {marks(row['cpp'])}")
```

- [ ] **Step 5: 跑测试确认通过**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_verify_drill.py -v`
Expected: 全 PASS。

- [ ] **Step 6: 手动 smoke**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/progress.py drill`
Expected: 输出 8 行 Week 1-8 的 STAR/Algo/Cpp 状态。当前实仓 STAR Week 全空 → 全 ✗；algorithm-drill.md 是新建模板 → 全 ✗。

- [ ] **Step 7: Commit**

```bash
git add notes/algorithm-drill.md study-plan/verify.py tests/test_verify_drill.py
git commit -m "feat: STAR / algorithm-drill markdown parsers"
```

---

### Task 11: `study-plan/run.py` daily driver

**Files:**
- Create: `study-plan/run.py`
- Create: `tests/test_run_driver.py`

run.py 是日级编排入口。子命令：`day N show/<phase>/done`、`week N show/check`、`today`、`next`。

- [ ] **Step 1: 写失败测试**

`tests/test_run_driver.py`：

```python
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "study_plan"


def setup_workspace(tmp_path):
    sp = tmp_path / "study-plan"
    sp.mkdir()
    shutil.copy(FIXTURES / "mini_progress.yaml", sp / "progress.yaml")
    # 把 run.py / progress.py / verify.py 复制过去（让 run.py import 找得到 verify）
    for fn in ("run.py", "progress.py", "verify.py"):
        src = REPO / "study-plan" / fn
        if src.exists():
            shutil.copy(src, sp / fn)
    return tmp_path


def run_driver(args, cwd):
    return subprocess.run(
        [sys.executable, "study-plan/run.py"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_day_show_includes_operator_and_artifacts(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["day", "2"], cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "Day 2" in out
    assert "row_softmax" in out
    assert "reference" in out  # phase or artifact label


def test_day_done_runs_strict_and_does_not_block(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["day", "2", "done"], cwd=tmp_path)
    # 软提示模式：未达标 strict 也返回 0
    assert res.returncode == 0


def test_today_picks_first_not_started(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["today"], cwd=tmp_path)
    assert res.returncode == 0
    assert "Day 1" in res.stdout or "day01" in res.stdout


def test_next_lists_next_pending_day(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["next"], cwd=tmp_path)
    assert res.returncode == 0


def test_day_out_of_range(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["day", "999"], cwd=tmp_path)
    assert res.returncode == 2


def test_week_show(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["week", "1"], cwd=tmp_path)
    assert res.returncode == 0
    assert "Week 1" in res.stdout or "week1" in res.stdout


def test_week_check_runs_all_three_sections(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["week", "1", "check"], cwd=tmp_path)
    assert res.returncode == 0
    out = res.stdout
    # 三段 header
    assert "strict verify" in out.lower() or "verify" in out.lower()
    assert "STAR" in out or "star" in out.lower()
    assert "drill" in out.lower() or "Algo" in out or "Cpp" in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_run_driver.py -v`
Expected: FAIL（run.py 不存在）。

- [ ] **Step 3: 写 `study-plan/run.py`**

```python
#!/usr/bin/env python3
"""study-plan daily driver.

Usage:
    python study-plan/run.py day N [show|reference|impl|tests|bench|profile|note|done]
    python study-plan/run.py week N [show|check]
    python study-plan/run.py today
    python study-plan/run.py next
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import verify  # noqa: E402

PROGRESS_FILE = HERE / "progress.yaml"
PHASE_ALIASES = {"impl": "implementation", "bench": "benchmark"}
PHASE_NAMES = {"reference", "implementation", "tests", "benchmark", "profile", "note"}
WEEKLY_DAY_NUMBERS = {7, 14, 21, 28, 35, 42, 49, 56}


def load_data() -> dict[str, Any]:
    import yaml
    return yaml.safe_load(PROGRESS_FILE.read_text(encoding="utf-8"))


def day_key(n: int) -> str:
    return f"day{n:02d}"


def find_day(data: dict, n: int) -> tuple[str, str, dict] | None:
    target = day_key(n)
    for week_key, week in data.items():
        if not week_key.startswith("week") or not isinstance(week, dict):
            continue
        if target in week:
            return week_key, target, week[target]
    return None


def cmd_day(args) -> int:
    data = load_data()
    found = find_day(data, args.n)
    if found is None:
        print(f"day {args.n} not found (valid: 0-56)", file=sys.stderr)
        return 2
    week_key, dk, day = found
    sub = args.sub or "show"

    if sub == "show":
        return _day_show(data, week_key, dk, day)
    if sub in PHASE_ALIASES or sub in PHASE_NAMES:
        phase = PHASE_ALIASES.get(sub, sub)
        return _day_phase(data, week_key, dk, day, phase)
    if sub == "done":
        return _day_done(data, week_key, dk, day, args.n)

    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


def _day_show(data, week_key, dk, day) -> int:
    op = day.get("operator", "")
    phase = day.get("phase", "")
    title = day.get("title", "")
    print(f"── Day {int(dk[3:])} · {op} · {phase} phase ──")
    print(f"title: {title}")
    print(f"date: {day.get('date') or '(unset)'}   status: {day.get('status')}   "
          f"daily_check: {day.get('daily_check', 0)}/3   "
          f"jd_tags: {' '.join(day.get('jd_tags', []))}")
    if op:
        op_data = data.get("operators", {}).get(op, {})
        kind = op_data.get("kind", "?")
        print(f"\noperator: {op} (kind={kind})")
        # 跑非 strict verify 拿真值
        try:
            results = verify.verify(
                data, target=("operator", op), strict=False, write=False,
                skip_tests=True, repo_root=REPO,
            )
            row = results["operators"][op]
            print("artifacts (non-strict):")
            for art in ["reference", "implementation", "tests", "benchmark", "profile", "note"]:
                ok = row["artifacts"].get(art, False)
                path = row["paths"].get(art, "?")
                mark = "✓" if ok else "✗"
                print(f"  {mark} {art:<18} {path}")
        except Exception as exc:
            print(f"(verify failed: {exc})")
    print("\ntoday's tasks:")
    for k, v in (day.get("tasks") or {}).items():
        mark = "☑" if v else "☐"
        print(f"  {mark} {k}")
    # next suggestion
    if op and phase:
        short = {"implementation": "impl", "benchmark": "bench"}.get(phase, phase)
        print(f"\nsuggested next:\n  → run.py day {int(dk[3:])} {short}")
    return 0


def _day_phase(data, week_key, dk, day, phase) -> int:
    op = day.get("operator")
    if not op:
        print(f"day {dk} has no operator", file=sys.stderr)
        return 2
    op_data = data["operators"][op]
    cmd = (op_data.get("commands") or {}).get(phase)
    if not cmd:
        cmd = _default_command(op, op_data, phase)
    if cmd is None:
        # editor open fallback
        path = verify.resolve_paths(op, op_data).get(_phase_to_path_key(phase))
        if path:
            full = REPO / path
            full.parent.mkdir(parents=True, exist_ok=True)
            if not full.exists() and phase == "note":
                full.write_text(_note_template(op), encoding="utf-8")
            editor = os.environ.get("EDITOR", "vi")
            print(f"$ {editor} {full}")
            return subprocess.call([editor, str(full)])
        print(f"no command for phase {phase}", file=sys.stderr)
        return 2
    print(f"$ {cmd}")
    rc = subprocess.call(cmd, shell=True, cwd=REPO)
    # 跑完跑非 strict verify 回写 yaml artifacts
    try:
        results = verify.verify(
            data, target=("operator", op), strict=False, write=False,
            skip_tests=True, repo_root=REPO,
        )
        verify.apply_results_in_memory(data, results)
        verify.write_back(data, results, PROGRESS_FILE)
    except Exception as exc:
        print(f"(post-phase verify skipped: {exc})")
    return rc


def _phase_to_path_key(phase: str) -> str:
    return {
        "reference": "impl",
        "implementation": "impl",
        "tests": "tests",
        "benchmark": "bench",
        "profile": "profile",
        "note": "note",
    }[phase]


def _default_command(op: str, op_data: dict, phase: str) -> str | None:
    paths = verify.resolve_paths(op, op_data)
    kind = op_data.get("kind", "reduction")
    if phase == "tests":
        return f"pytest {paths['tests']} -v"
    if phase == "benchmark":
        return f"python {paths['bench']}"
    if phase == "profile":
        if kind in {"reduction", "elementwise", "quant"}:
            return f"bash scripts/run_ncu.sh {op}"
        if kind in {"attention", "gemm"}:
            return f"bash scripts/run_nsys.sh {op}"
    if phase in {"reference", "implementation", "note"}:
        return None  # editor fallback
    return None


def _note_template(op: str) -> str:
    return f"""# {op}

## Result
- baseline:
- target:
- final:

## Bottleneck
（哪一段是热点；roofline 大致位置；memory-bound 还是 compute-bound）

## Next experiment
（下一步打算改什么；预计影响哪一项指标）
"""


def _day_done(data, week_key, dk, day, n) -> int:
    print(f"── Day {n} done: strict verify ──")
    op = day.get("operator")
    if op:
        results = verify.verify(
            data, target=("operator", op), strict=True, write=False,
            skip_tests=False, repo_root=REPO,
        )
        verify.write_back(data, results, PROGRESS_FILE)
        verify.print_summary(results, strict=True)
    # Day 1 gate: 顺带 strict verify 4 个 warmup
    if n in (0, 1):
        print("\n── Day 1 gate: warmup operators ──")
        warmup = ["vector_add", "axpy", "row_sum", "row_max"]
        for wo in warmup:
            if wo in data.get("operators", {}):
                wres = verify.verify(
                    data, target=("operator", wo), strict=True, write=False,
                    skip_tests=True, repo_root=REPO,
                )
                row = wres["operators"][wo]
                count = sum(1 for v in row["artifacts"].values() if v)
                print(f"  {wo}: {count}/6")
    print("\n填 daily_check (0-3) / weaknesses / next_fix 后保存。")
    return 0


def cmd_week(args) -> int:
    data = load_data()
    if args.sub == "check":
        return _week_check(data, args.n)
    return _week_show(data, args.n)


def _week_show(data, n) -> int:
    week = data.get(f"week{n}")
    if not week:
        print(f"week {n} not found", file=sys.stderr)
        return 2
    print(f"── Week {n} ──")
    for dk in sorted(week.keys()):
        d = week[dk]
        print(f"  {dk}  {d.get('status', '?'):<14}  {d.get('operator', '-'):<14}  "
              f"{d.get('phase', '-'):<14}  {d.get('title', '')}")
    return 0


def _week_check(data, n) -> int:
    print(f"── Week {n} Check ──")
    print("\n[1/3] strict verify (per day)")
    week = data.get(f"week{n}", {})
    for dk in sorted(week.keys()):
        d = week[dk]
        op = d.get("operator")
        if not op:
            continue
        res = verify.verify(
            data, target=("operator", op), strict=True, write=False,
            skip_tests=True, repo_root=REPO,
        )
        row = res["operators"][op]
        passed = sum(1 for v in row["artifacts"].values() if v)
        print(f"  {dk}  {op:<14}  {passed}/6")

    print("\n[2/3] STAR weekly")
    star_path = REPO / "notes" / "star-weekly.md"
    star_ok = verify.check_star_filled(star_path, week=n, strict=False)
    print(f"  Week {n}: {'✓' if star_ok else '✗ (notes/star-weekly.md Week ' + str(n) + ' 段为空)'}")

    print("\n[3/3] algorithm / cpp drill")
    drill_path = REPO / "notes" / "algorithm-drill.md"
    algo_ok = verify.check_drill_done(drill_path, week=n, kind="algo")
    cpp_ok = verify.check_drill_done(drill_path, week=n, kind="cpp")
    print(f"  Algo: {'✓' if algo_ok else '✗'}")
    print(f"  Cpp:  {'✓' if cpp_ok else '✗'}")

    # 写回周检日字段（star_filled / algo_drill_done / cpp_drill_done）
    weekly_dk = day_key(n * 7)
    if weekly_dk in week:
        week[weekly_dk]["star_filled"] = star_ok
        week[weekly_dk]["algo_drill_done"] = algo_ok
        week[weekly_dk]["cpp_drill_done"] = cpp_ok
        import yaml
        PROGRESS_FILE.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )

    if n in (2, 4, 6, 8):
        print(f"\n阶段检: 请填 day{n*7}.stage_check_score (0-100)")
    return 0


def cmd_today(args) -> int:
    data = load_data()
    for n in range(1, 57):
        found = find_day(data, n)
        if found and found[2].get("status") == "not_started":
            return _day_show(data, found[0], found[1], found[2])
    print("All 56 days are started.")
    return 0


def cmd_next(args) -> int:
    data = load_data()
    for n in range(1, 57):
        found = find_day(data, n)
        if found and found[2].get("status") == "not_started":
            print(f"next: day {n} ({found[2].get('title', '')})")
            return 0
    print("no pending days")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="run")
    sub = parser.add_subparsers(dest="cmd")

    p_day = sub.add_parser("day")
    p_day.add_argument("n", type=int)
    p_day.add_argument("sub", nargs="?", default=None)

    p_week = sub.add_parser("week")
    p_week.add_argument("n", type=int)
    p_week.add_argument("sub", nargs="?", default="show")

    sub.add_parser("today")
    sub.add_parser("next")

    args = parser.parse_args()

    if args.cmd == "day":
        return cmd_day(args)
    if args.cmd == "week":
        return cmd_week(args)
    if args.cmd == "today":
        return cmd_today(args)
    if args.cmd == "next":
        return cmd_next(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python -m pytest tests/test_run_driver.py -v`
Expected: 全 PASS。如果 `test_today_picks_first_not_started` 失败，检查 mini_progress.yaml 里 day01 是否真是 `not_started`（fixture 里应当是）。

- [ ] **Step 5: 手动 smoke**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py today`
Expected: 输出 Day 1 信息屏（仓库校准 + Nsight Compute WSL2 验证）。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py day 5`
Expected: row_softmax · profile phase 信息屏，artifacts 全 ✗。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py week 1`
Expected: Week 1 七天列表。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py week 1 check`
Expected: 三段汇总；STAR/Algo/Cpp 全 ✗（实仓未填）；day07 yaml 字段被回写为 false。

- [ ] **Step 6: Commit**

```bash
git add study-plan/run.py tests/test_run_driver.py
git commit -m "feat: study-plan/run.py daily driver with day/week/today/next subcommands"
```

---

### Task 12: 文档同步 + 04_verify_all.sh + Phase B 收尾

**Files:**
- Modify: `README.md`
- Modify: `study-plan/README.md`
- Modify: `AGENTS.md`
- Modify: `scripts/04_verify_all.sh`

- [ ] **Step 1: 把根 README 入口换成 driver**

打开 `README.md`，把开头几段换成（保留 8 周计划链接）：

```markdown
# llm-kernel-lab

LLM 推理框架 / 加速学习仓库。每天的入口是 `study-plan/run.py`：

```bash
python study-plan/run.py today          # 看今天该做什么
python study-plan/run.py day 5 show     # 看 Day 5 的状态
python study-plan/run.py day 5 done     # 跑 strict verify，更新 yaml
python study-plan/run.py week 1 check   # 周检：strict + STAR + drill
```

进度真理源：`study-plan/progress.yaml`。所有 artifact 字段由 `verify` 引擎写回，**不要手动改**；要改状态就去补真文件，然后 `run.py done` 触发 verify。

8 周计划：[study-plan/inference-acceleration-plan.md](study-plan/inference-acceleration-plan.md)。
```

不要删除 README 里其它有用内容（环境、目录结构）；只换开头入口段。

- [ ] **Step 2: study-plan/README.md 加 driver 一节**

在 `study-plan/README.md` 末尾追加：

```markdown
## Daily driver

```bash
python study-plan/run.py day N                  # 信息屏
python study-plan/run.py day N tests            # 跑当天 operator 的 pytest
python study-plan/run.py day N bench            # 跑当天 operator 的 benchmark
python study-plan/run.py day N profile          # 跑当天 operator 的 ncu/nsys
python study-plan/run.py day N done             # strict verify + 写回 yaml
python study-plan/run.py week N check           # 周检：strict + STAR + drill
```

## Verify 引擎

```bash
python study-plan/progress.py verify --operator row_softmax --strict
python study-plan/progress.py verify --all --write
python study-plan/progress.py drill              # STAR / algo / cpp drill 总览
python study-plan/progress.py status --day 5     # 单日详细状态
```

`progress.yaml` 是真理源；artifact / status / 周检 hook 字段由 verify 写回。
用户笔记字段（daily_check, weaknesses, next_fix, notes, weekly_check_score）可手动改，
也可走 dashboard 编辑界面。
```

- [ ] **Step 3: AGENTS.md 加 verify 命令**

定位 `AGENTS.md` 的 "Required Verification" 段（在 `bash scripts/04_verify_all.sh` 那行附近）。在末尾追加：

```markdown
After changing operator artifacts (kernel / test / benchmark / profile / note), run:

```bash
python study-plan/progress.py verify --operator <op>
```

Then commit progress.yaml diff together with the artifact change.
```

- [ ] **Step 4: scripts/04_verify_all.sh 末尾加测试**

打开 `scripts/04_verify_all.sh`，在文件末尾（最后一个命令之后）追加：

```bash

# study-plan driver / verify 引擎测试
pytest tests/test_verify.py tests/test_progress_cli.py tests/test_run_driver.py \
       tests/test_verify_drill.py tests/test_study_plan_dashboard.py -q
```

注意保持 `set -euo pipefail` 已经在文件开头。

- [ ] **Step 5: 跑 04_verify_all.sh 全过**

Run: `bash scripts/04_verify_all.sh`
Expected: 现有 verification + 新增 pytest 全 PASS。

如果有失败：
- 测试自身失败 → 回到对应 Task 修。
- ncu / 真 GPU 测试因环境失败 → 这是 study-plan/run.py 之外的事，**不修**；记录后继续。

- [ ] **Step 6: 全量人工验收 — Phase B**

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py today`
Expected: Day 1 信息屏。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py day 5 show`
Expected: 输出符合 spec §4 的模板（标题、date、status、artifacts 6 行、tasks、suggested next）。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py day 1 done`
Expected: 跑 strict verify；warmup 段列出 vector_add 6/6、axpy ≤5/6、row_sum 6/6、row_max ≤5/6；不阻塞。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/run.py week 1 check`
Expected: 三段输出顺序为 verify → STAR → drill；day07.star_filled / algo_drill_done / cpp_drill_done 在 yaml 里被回写。

Run: `~/miniconda3/envs/llm-kernel-lab/bin/python study-plan/dashboard.py --build`
Expected: 重新生成 HTML，新字段（operator/phase/drill chip）正常显示。

- [ ] **Step 7: 决定是否提交 progress.yaml 的回写**

`run.py week 1 check` 把 `day07.star_filled` 等字段写成了 false（实仓现状）。`git diff study-plan/progress.yaml`：
- 只有这三个字段变 false → 提交：`git commit -am "chore: verify drill state"`
- 还有别的字段被改 → 回滚 `git checkout study-plan/progress.yaml` 然后排查。

- [ ] **Step 8: Commit 文档同步**

```bash
git add README.md study-plan/README.md AGENTS.md scripts/04_verify_all.sh
git commit -m "docs: redirect entry to study-plan/run.py + verify CLI"
```

至此 Phase B 完成。

---

## 验收对照（来自 spec §8）

**Phase A：**

- [x] `progress.py verify --operator row_sum --strict` 输出 6/6 — Task 9 Step 2
- [x] `progress.py verify --operator axpy --strict` 输出 ≤5/6 + 缺失说明 — Task 9 Step 2
- [x] `progress.py verify --operator row_softmax` 输出 0/6 — Task 9 Step 2
- [x] `progress.py verify --all --write` 后 git diff 只动 verify 字段 — Task 9 Step 2
- [x] `dashboard.py --build` 渲染新字段不报错 — Task 9 Step 3
- [x] `tests/test_verify.py` 全过 — Task 4 Step 9 / Task 9 Step 1

**Phase B：**

- [x] `run.py today` 推断当前应做的 day — Task 11 Step 5
- [x] `run.py day 5 show` 输出符合 spec §4 模板 — Task 12 Step 6
- [x] `run.py day 1 done` 报 warmup 算子未达 strict — Task 12 Step 6
- [x] `run.py week 1 check` 三段汇总按顺序 — Task 12 Step 6
- [x] `bash scripts/04_verify_all.sh` 末尾 pytest 全过 — Task 12 Step 5
- [x] 根 README 入口指向 driver — Task 12 Step 1

---

## Self-Review notes（已在写计划时处理）

1. **Spec coverage check** — spec §1-§8 全部映射到 Task 0-12，无遗漏。`star_filled`/`algo_drill_done`/`cpp_drill_done` 在 Task 5（migration）+ Task 10（解析）+ Task 11（_week_check 写回）三处协同。
2. **Placeholder scan** — 计划里所有"实现……"都给了完整代码块；模板 / fixture 全部写出。唯一一处 placeholder 是 Task 7 Step 4 的 dashboard 模板嵌入（"具体 HTML 字符串怎么织取决于现有写法"），因为 `dashboard.py` 当前模板形态未在计划里直接 dump，让执行者按现有 pattern 织入；这属于"按现有 pattern"而非"自由发挥"。
3. **Type / 名字一致性** — 函数名跨任务核对：`verify.verify()`, `verify.verify_day()`, `verify.write_back()`, `verify.apply_results_in_memory()`, `verify.resolve_paths()`, `verify.print_summary()`, `verify.print_day_status()`, `verify.print_drill_summary()`, `verify.collect_drill_summary()`, `verify.check_star_filled()`, `verify.check_drill_done()`, `verify.UnknownTargetError`, `verify.MissingDependencyError`, `verify.BackupError`。Task 2-4 定义，Task 6/7/10/11 调用，名字一致。
4. **Spec 偏离** — 计划开头明确声明放弃 ruamel.yaml；其它选择和 spec 对齐。

