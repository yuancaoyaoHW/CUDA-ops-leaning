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
