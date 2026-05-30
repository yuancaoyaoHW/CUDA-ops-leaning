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


# ---- check_implementation / check_reference ---------------------------------


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
    assert verify.check_reference(op, paths, strict=False) is True  # file exists
    assert verify.check_reference(op, paths, strict=True) is False

    f.write_text("import torch\nimport triton\n")
    assert verify.check_reference(op, paths, strict=True) is True

    # explicit reference_inline also counts
    f.write_text("import triton\n")
    op2 = {"kind": "reduction", "reference_inline": True}
    assert verify.check_reference(op2, paths, strict=True) is True


# ---- check_benchmark --------------------------------------------------------


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
    # non-strict only checks file existence
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


# ---- check_profile ----------------------------------------------------------


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
    # any hit = non-strict pass
    assert verify.check_profile(paths, th, strict=False) is True
    # strict needs at least one >= min_size
    assert verify.check_profile(paths, th, strict=True) is True
    # remove big, only small remains, strict should fail
    big.unlink()
    assert verify.check_profile(paths, th, strict=True) is False


# ---- check_note --------------------------------------------------------------


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
    # missing next_experiment section
    assert verify.check_note(paths, th, strict=True) is False


# ---- check_tests -------------------------------------------------------------


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
    # skip_tests=True means strict also fails
    assert verify.check_tests("x", op, paths, th, strict=True, repo_root=tmp_path,
                              skip_tests=True) is False


# ---- Task 4: verify() top-level + derive_status + yaml writeback -----------

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
