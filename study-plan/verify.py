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


# ---- exceptions -------------------------------------------------------------


class UnknownTargetError(Exception):
    """Raised when a --day or --operator target doesn't exist in the data."""
    pass


class MissingDependencyError(Exception):
    """Raised when a required external dependency is unreachable."""
    pass


class BackupError(Exception):
    """Raised when the .bak write fails."""
    pass


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

    overrides = (op.get("thresholds") or {})

    return Thresholds(
        note_min_lines=overrides.get("note_min_lines", defaults.get("note_min_lines", 30)),
        note_required_sections=overrides.get(
            "note_required_sections",
            defaults.get("note_required_sections", ["bottleneck", "next_experiment"]),
        ),
        bench_required_keys=overrides.get("bench_required_keys", bench_keys_default),
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


# ---- check_* ---------------------------------------------------------------


def check_implementation(paths: Paths, *, strict: bool) -> bool:
    """Non-strict: impl file exists. Strict: file exists AND non-empty."""
    if not paths.impl.exists():
        return False
    if strict and paths.impl.stat().st_size == 0:
        return False
    return True


def check_reference(op: dict[str, Any], paths: Paths, *, strict: bool) -> bool:
    """Non-strict: impl file exists. Strict: grep for torch/pytorch_reference or inline flag."""
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
    """Non-strict: test file exists. Strict: run pytest, exit 0."""
    if not paths.tests.exists():
        return False
    if not strict:
        return True
    if skip_tests:
        return False
    overrides: dict[str, str] = (op.get("commands") or {})
    if "tests" in overrides:
        cmd = overrides["tests"]
    else:
        try:
            rel = paths.tests.relative_to(repo_root)
        except ValueError:
            rel = paths.tests
        cmd = f"pytest {rel} -v"
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
    """Non-strict: bench_json exists. Strict: JSON valid + required keys present and non-null."""
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
    """Non-strict: glob hits any file. Strict: glob hits AND any match >= min size."""
    matches: list[Path] = []
    for g in paths.profile_globs:
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
    """Non-strict: file exists. Strict: line count + required sections."""
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
        # normalize: underscores in config match spaces in headings
        needle = section.lower().replace("_", " ")
        if needle not in lower:
            return False
    return True


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
    """Derive operator status from artifact booleans (spec section 3)."""
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
    try:
        bak = yaml_path.with_suffix(yaml_path.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(yaml_path.read_bytes())
    except OSError as exc:
        raise BackupError(f"backup failed: {exc}") from exc


def _resolve_targets(target: tuple, data: dict[str, Any]) -> list[str]:
    if target[0] == "all":
        return sorted((data.get("operators") or {}).keys())
    if target[0] == "operator":
        op = target[1]
        if op not in (data.get("operators") or {}):
            raise UnknownTargetError(f"unknown operator: {op}")
        return [op]
    if target[0] == "day":
        # day target maps to its operator(s)
        week_key, day_key = _find_day(data, target[1])
        if week_key is None:
            raise UnknownTargetError(f"unknown day: {target[1]}")
        day = data[week_key][day_key] or {}
        if "operators" in day:
            return list(day["operators"])
        if "operator" in day:
            return [day["operator"]]
        return []  # review-only day, valid but no operators
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
    yaml_path: Path | None = None,
    data: dict[str, Any] | None = None,
    repo_root: Path,
    target: tuple,
    strict: bool,
    write: bool,
    skip_tests: bool = False,
) -> VerifyResult:
    """Top-level verify orchestrator.

    Resolves targets, runs check_* for each operator, derives status,
    and optionally writes back artifacts + status to the YAML file.

    Either yaml_path or data must be provided. If write=True, yaml_path
    is required (to know where to write back).
    """
    if data is None:
        if yaml_path is None:
            raise ValueError("either yaml_path or data must be provided")
        data = _load_yaml(yaml_path)
    if write and yaml_path is None:
        raise ValueError("yaml_path required when write=True")

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


# ---- helpers for CLI wrappers ------------------------------------------------


def apply_results_in_memory(data: dict[str, Any], results: VerifyResult) -> None:
    """Write verify results into in-memory dict (no disk)."""
    operators = data.setdefault("operators", {})
    for op_name, res in results.operators.items():
        if op_name in operators and operators[op_name] is not None:
            operators[op_name]["artifacts"] = res.artifacts
            operators[op_name]["status"] = res.status


def write_back(data: dict[str, Any], results: VerifyResult, yaml_path: Path) -> None:
    """Apply results to data and write to disk with backup."""
    try:
        _backup(yaml_path)
    except BackupError:
        raise
    except OSError as exc:
        raise BackupError(f"backup failed: {exc}") from exc
    apply_results_in_memory(data, results)
    _save_yaml(yaml_path, data)


def verify_day(
    data: dict[str, Any],
    day_num: int,
    *,
    repo_root: Path,
    strict: bool = False,
    skip_tests: bool = True,
) -> VerifyResult:
    """Convenience wrapper: verify a single day's operators."""
    return verify(
        data=data,
        repo_root=repo_root,
        target=("day", day_num),
        strict=strict,
        write=False,
        skip_tests=skip_tests,
    )


def print_summary(results: VerifyResult, *, strict: bool) -> None:
    """Print one-line-per-operator summary to stdout."""
    label = "strict" if strict else "non-strict"
    print(f"verify ({label}):")
    short = {"reference": "ref", "implementation": "impl", "tests": "tests",
             "benchmark": "bench", "profile": "profile", "note": "note"}
    for op_name in sorted(results.operators):
        res = results.operators[op_name]
        score = sum(1 for v in res.artifacts.values() if v)
        marks = " ".join(
            f"{'✓' if res.artifacts.get(k) else '✗'} {short[k]}"
            for k in ARTIFACT_ORDER
        )
        print(f"  {op_name:<22} {score}/6  {marks}   [{res.status}]")


def print_day_status(result: VerifyResult, day_num: int) -> None:
    """Print a single day's operator status."""
    day_key = f"day{day_num:02d}"
    if not result.operators:
        print(f"{day_key}: review-only day (no operators)")
        return
    print(f"{day_key}:")
    for op_name in sorted(result.operators):
        res = result.operators[op_name]
        score = sum(1 for v in res.artifacts.values() if v)
        print(f"  {op_name}: {score}/6 [{res.status}]")


# ---- Task 10 stubs ----------------------------------------------------------


def collect_drill_summary(data: dict[str, Any], *, repo_root: Path) -> dict:
    """Stub: will be implemented in Task 10."""
    return {"_stub": True}


def print_drill_summary(summary: dict) -> None:
    """Stub: will be implemented in Task 10."""
    if summary.get("_stub"):
        print("drill summary not implemented yet (Task 10)")
