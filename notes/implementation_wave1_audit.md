# Implementation Wave 1 Audit Checklist

Purpose: verify this continuation wave without touching operator code. The wave is limited to benchmark JSON alignment for existing warmup operators and Day 8-14 guides.

## Current Wave Scope

- Benchmark JSON worker: align existing warmup operators so each benchmark can emit or maintain verifier-readable JSON under `reports/json/`.
- Warmup operators in scope: `vector_add`, `axpy`, `row_sum`, and `row_max`.
- Day 8-14 guide worker: create or refresh active guide notes for `day08` through `day14` from the current `study-plan/progress.yaml` week2 plan.
- No Triton kernel, CUDA extension, test behavior, or operator implementation work belongs in this wave.

## Allowed Files

Benchmark JSON worker may touch only:

- `benchmarks/bench_vector_add.py`
- `benchmarks/bench_axpy.py`
- `benchmarks/bench_row_sum.py`
- `benchmarks/bench_row_max.py`
- `benchmarks/bench_utils.py`, only if shared JSON writing removes duplication
- `reports/json/vector_add_bench.json`
- `reports/json/axpy_bench.json`
- `reports/json/row_sum_bench.json`
- `reports/json/row_max_bench.json`

Day 8-14 guide worker may touch only active guide files for the current week2 plan:

- `study-plan/week2/day08-fused-add-rmsnorm.md`
- `study-plan/week2/day09-swiglu-gelu.md`
- `study-plan/week2/day10-rope.md`
- `study-plan/week2/day11-rmsnorm-cuda-ext.md`
- `study-plan/week2/day12-rmsnorm-cuda-ext-benchmark.md`
- `study-plan/week2/day13-kv-cache-append.md`
- `study-plan/week2/day14-review-stage-check.md`

This audit note itself is allowed to exist at `notes/implementation_wave1_audit.md`.

## Files Not To Touch

- Unrelated dirty files shown by `git status --short`.
- `study-plan/progress.yaml`, unless a later operator artifact verification task explicitly runs the verifier with write-back and commits that verifier diff together with the artifact change.
- `reports/ncu/` and `reports/nsys/` GPU profile reports, unless a worker is explicitly assigned to generate a profile.
- Operator code under `kernels/`.
- Operator tests under `tests/`, except read-only inspection.
- Archived guide drafts under `study-plan/archive/`, except read-only inspection.
- Setup scripts and environment files.

## Benchmark JSON Verification

Smallest relevant checks before running GPU benchmarks:

```bash
python -m pytest tests/test_verify.py tests/test_progress_cli.py -q
python study-plan/progress.py verify --operator vector_add
python study-plan/progress.py verify --operator axpy
python study-plan/progress.py verify --operator row_sum
python study-plan/progress.py verify --operator row_max
```

If benchmark scripts are changed and GPU access is available, run only the affected benchmark scripts:

```bash
python benchmarks/bench_vector_add.py
python benchmarks/bench_axpy.py
python benchmarks/bench_row_sum.py
python benchmarks/bench_row_max.py
```

Then validate JSON syntax and verifier visibility:

```bash
python -m json.tool reports/json/vector_add_bench.json >/dev/null
python -m json.tool reports/json/axpy_bench.json >/dev/null
python -m json.tool reports/json/row_sum_bench.json >/dev/null
python -m json.tool reports/json/row_max_bench.json >/dev/null
python study-plan/progress.py verify --operator vector_add --strict --skip-tests
python study-plan/progress.py verify --operator axpy --strict --skip-tests
python study-plan/progress.py verify --operator row_sum --strict --skip-tests
python study-plan/progress.py verify --operator row_max --strict --skip-tests
```

Expected JSON alignment:

- File path follows `reports/json/<operator>_bench.json`.
- JSON is valid and includes non-empty top-level `gbps`, because `study-plan/verify.py` checks `bench_required_keys_default`.
- Detailed per-shape rows may be included, but the verifier-visible top-level key must remain present.
- No `reports/ncu/` or `reports/nsys/` files are created by this worker.

## Day 8-14 Guide Verification

Read-only source of truth:

```bash
sed -n '478,675p' study-plan/progress.yaml
```

After guide edits, verify the active files exist and match the week2 operators:

```bash
test -f study-plan/week2/day08-fused-add-rmsnorm.md
test -f study-plan/week2/day09-swiglu-gelu.md
test -f study-plan/week2/day10-rope.md
test -f study-plan/week2/day11-rmsnorm-cuda-ext.md
test -f study-plan/week2/day12-rmsnorm-cuda-ext-benchmark.md
test -f study-plan/week2/day13-kv-cache-append.md
test -f study-plan/week2/day14-review-stage-check.md
rg -n "fused_add_rmsnorm|swiglu|gelu|rope|rmsnorm_cuda_ext|kv_cache_append|stage_check" study-plan/week2
python study-plan/progress.py status --day 8
python study-plan/progress.py status --day 14
```

Guide content checklist:

- Day 8 maps to `fused_add_rmsnorm`.
- Day 9 maps to `swiglu` and includes `gelu` as the paired activation work.
- Day 10 maps to `rope`.
- Day 11 and Day 12 both map to `rmsnorm_cuda_ext`, with Day 11 focused on implementation/build path and Day 12 focused on tests/benchmark.
- Day 13 maps to `kv_cache_append`.
- Day 14 is a review day and must not invent an operator artifact.
- Each guide names allowed outputs and narrow verification without requiring expensive GPU profile runs.

## Final Integration Checklist

Run these before handing back the wave:

```bash
git diff --cached
git status --short
python -m pytest tests/test_verify.py tests/test_progress_cli.py -q
rg -n "benchmark JSON|Day 8-14|progress.yaml|verification|git status" notes/implementation_wave1_audit.md
```

Targeted tests by worker:

- Benchmark JSON worker: run `python -m pytest tests/test_verify.py tests/test_progress_cli.py -q`, JSON syntax checks, and `python study-plan/progress.py verify --operator <op> --strict --skip-tests` for touched operators.
- Day 8-14 guide worker: run the file existence checks, the week2 `rg`, and `python study-plan/progress.py status --day 8` plus `--day 14`.
- Do not run `bash scripts/04_verify_all.sh` for this audit-only wave unless the scope expands to Triton kernels or operator tests.

## Risks And Rollback Notes

- Risk: strict verifier can fail for reasons unrelated to benchmark JSON, such as missing profile or note artifacts. Record the failing artifact instead of expanding scope.
- Risk: benchmark JSON schema can drift from `study-plan/verify.py`; keep the top-level `gbps` key even if richer `results` rows are added.
- Risk: guide filenames may diverge from future naming conventions. Keep names stable for Day 8-14 in this wave and do not rewrite archived drafts.
- Risk: GPU benchmarks may be unavailable or too expensive on the current machine. In that case, run syntax and verifier checks only, then state that benchmark execution was skipped.
- Rollback for benchmark JSON changes: revert only the touched benchmark script or JSON file; leave kernels, tests, progress, and profiles unchanged.
- Rollback for guide changes: remove or revert only the active `study-plan/week2/day08` through `day14` guide files created by this wave.
