from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def load_dashboard_module():
    module_path = Path(__file__).resolve().parents[1] / "study-plan" / "dashboard.py"
    spec = importlib.util.spec_from_file_location("study_plan_dashboard", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_sample_progress(path: Path) -> None:
    data = {
        "meta": {"title": "Test Plan"},
        "operators": {
            "row_softmax": {
                "status": "not_started",
                "artifacts": {
                    "reference": False,
                    "implementation": False,
                    "tests": False,
                    "benchmark": False,
                    "profile": False,
                    "note": False,
                },
                "notes": "",
            }
        },
        "gpu_libraries": {
            "cutlass": {"status": "not_started", "evidence": []},
        },
        "week1": {
            "day01": {
                "title": "Day One",
                "date": "",
                "status": "not_started",
                "daily_check": 0,
                "jd_tags": ["kernel"],
                "tasks": {"audit": False, "plan": False},
                "artifacts": {"docs": False},
                "verification": "",
                "weaknesses": "",
                "next_fix": "",
                "notes": "",
            }
        },
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_update_day_writes_extended_progress_fields(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    ok = dashboard.update_day(
        1,
        {
            "status": "done",
            "date": "2026-05-30",
            "daily_check": 3,
            "tasks": {"audit": True, "plan": True, "unknown": True},
            "artifacts": {"docs": True},
            "verification": "python study-plan/progress.py analyze",
            "weaknesses": "none",
            "next_fix": "continue",
            "notes": "updated",
        },
    )

    assert ok is True
    saved = yaml.safe_load(progress_file.read_text(encoding="utf-8"))
    day = saved["week1"]["day01"]
    assert day["status"] == "done"
    assert day["daily_check"] == 3
    assert day["tasks"] == {"audit": True, "plan": True}
    assert day["artifacts"] == {"docs": True}
    assert day["verification"] == "python study-plan/progress.py analyze"
    assert day["weaknesses"] == "none"
    assert day["next_fix"] == "continue"
    assert day["notes"] == "updated"


def test_update_day_derives_status_from_checklists(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    assert dashboard.update_day(
        1,
        {
            "status": "done",
            "tasks": {"audit": True},
            "artifacts": {"docs": False},
        },
    )

    saved = yaml.safe_load(progress_file.read_text(encoding="utf-8"))
    assert saved["week1"]["day01"]["status"] == "in_progress"


def test_update_day_preserves_blocked_status(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    assert dashboard.update_day(
        1,
        {
            "status": "blocked",
            "tasks": {"audit": False, "plan": False},
            "artifacts": {"docs": False},
        },
    )

    saved = yaml.safe_load(progress_file.read_text(encoding="utf-8"))
    assert saved["week1"]["day01"]["status"] == "blocked"


def test_update_operator_and_gpu_library_write_yaml(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    assert dashboard.update_operator(
        "row_softmax",
        {
            "status": "benchmark_stage",
            "artifacts": {"reference": True, "implementation": True, "tests": True},
            "notes": "correctness complete",
        },
    )
    assert dashboard.update_gpu_library(
        "cutlass",
        {
            "status": "in_progress",
            "evidence": ["reports/cutlass/gemm.md"],
        },
    )

    saved = yaml.safe_load(progress_file.read_text(encoding="utf-8"))
    op = saved["operators"]["row_softmax"]
    assert op["status"] == "benchmark_stage"
    assert op["artifacts"]["reference"] is True
    assert op["artifacts"]["implementation"] is True
    assert op["artifacts"]["tests"] is True
    assert op["artifacts"]["benchmark"] is False
    assert op["notes"] == "correctness complete"
    assert saved["gpu_libraries"]["cutlass"] == {
        "status": "in_progress",
        "evidence": ["reports/cutlass/gemm.md"],
    }


def test_get_api_data_exposes_editable_sections(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    api_data = dashboard.get_api_data()

    assert api_data["weeks"][0]["days"][0]["artifacts"] == {"docs": False}
    assert api_data["weeks"][0]["days"][0]["jd_tags"] == ["kernel"]
    assert api_data["weeks"][0]["days"][0]["task_total"] == 2
    assert api_data["summary"]["total_days"] == 56
    assert api_data["current_day"]["num"] == 1
    assert "row_softmax 还没有闭环" in api_data["risks"][0]
    assert "row_softmax" in api_data["operators"]
    assert "cutlass" in api_data["gpu_libraries"]


def test_render_dashboard_uses_external_assets_and_accessible_dialog(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)

    html = dashboard.render_dashboard(embed_data=False)

    assert 'href="dashboard.css"' in html
    assert 'src="dashboard.js"' in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert 'id="initial-data"' not in html
