from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_verify_module():
    import sys

    module_path = Path(__file__).resolve().parents[1] / "study-plan" / "verify.py"
    spec = importlib.util.spec_from_file_location("study_plan_verify", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["study_plan_verify"] = module
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
