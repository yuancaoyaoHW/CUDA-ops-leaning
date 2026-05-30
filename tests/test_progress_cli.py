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
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPO / "study-plan"),
        },
    )


def setup_workspace(tmp_path):
    sp = tmp_path / "study-plan"
    sp.mkdir()
    shutil.copy(FIXTURES / "mini_progress.yaml", sp / "progress.yaml")
    (tmp_path / "kernels" / "triton" / "row_softmax").mkdir(parents=True)
    (tmp_path / "kernels" / "triton" / "row_softmax" / "row_softmax.py").write_text(
        "import torch\n\ndef pytorch_reference(x): return torch.softmax(x, -1)\n"
    )
    return sp


def test_verify_cli_default_all(tmp_path):
    setup_workspace(tmp_path)
    result = run_progress(["verify"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "row_softmax" in result.stdout


def test_verify_cli_operator(tmp_path):
    setup_workspace(tmp_path)
    result = run_progress(["verify", "--operator", "row_softmax"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "row_softmax" in result.stdout


def test_verify_cli_unknown_operator_exits_2(tmp_path):
    setup_workspace(tmp_path)
    result = run_progress(["verify", "--operator", "no_such_op"], cwd=tmp_path)
    assert result.returncode == 2
    assert "no_such_op" in result.stderr or "no_such_op" in result.stdout


def test_status_cli_day(tmp_path):
    setup_workspace(tmp_path)
    result = run_progress(["status", "--day", "2"], cwd=tmp_path)
    # day02 doesn't exist in mini fixture, but day01 and day07 do
    # Let's test with day 1 which exists
    result = run_progress(["status", "--day", "1"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "day01" in result.stdout or "Day 1" in result.stdout


def test_legacy_subcommands_still_work(tmp_path):
    setup_workspace(tmp_path)
    result = run_progress(["analyze"], cwd=tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "JD" in result.stdout or "标签" in result.stdout or "分析" in result.stdout
