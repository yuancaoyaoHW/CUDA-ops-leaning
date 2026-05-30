from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "study_plan"


def load_verify():
    spec = importlib.util.spec_from_file_location("verify", REPO / "study-plan" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = mod
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
