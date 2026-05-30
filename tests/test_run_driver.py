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
    res = run_driver(["day", "1"], cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "Day 1" in out
    assert "row_softmax" in out
    assert "reference" in out


def test_day_done_runs_strict_and_does_not_block(tmp_path):
    setup_workspace(tmp_path)
    res = run_driver(["day", "1", "done"], cwd=tmp_path)
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
    assert "strict verify" in out.lower() or "verify" in out.lower()
    assert "STAR" in out or "star" in out.lower()
    assert "drill" in out.lower() or "Algo" in out or "Cpp" in out
