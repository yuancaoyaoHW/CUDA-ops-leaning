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


def test_render_dashboard_uses_react_static_index(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        '<!doctype html><div id="root"></div><script type="module" src="/assets/index.js"></script>',
        encoding="utf-8",
    )
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)
    monkeypatch.setattr(dashboard, "STATIC_DIR", static_dir)
    monkeypatch.setattr(dashboard, "STATIC_INDEX", static_dir / "index.html")

    html = dashboard.render_dashboard()

    assert 'id="root"' in html
    assert "/assets/index.js" in html
    assert 'id="initial-data"' not in html


def test_load_guide_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    result = dashboard.load_guide(1)

    assert result is None


def test_load_guide_returns_parsed_yaml(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "day01.yaml").write_text(
        "day: 1\ntasks:\n  audit:\n    summary: Check files\n    steps:\n      - list dir\n    done_when: audit.md exists\n    time_minutes: 20\n    depends_on: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    result = dashboard.load_guide(1)

    assert result is not None
    assert result["day"] == 1
    assert result["tasks"]["audit"]["summary"] == "Check files"
    assert result["tasks"]["audit"]["steps"] == ["list dir"]
    assert result["tasks"]["audit"]["time_minutes"] == 20


def test_enrich_day_merges_guide(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "day01.yaml").write_text(
        "day: 1\ntasks:\n  audit:\n    summary: Check files\n    steps:\n      - list dir\n    done_when: audit.md exists\n    time_minutes: 20\n    depends_on: []\ntotal_time_minutes: 20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    day = {"num": 1, "week": 1, "tasks": {"audit": False}, "artifacts": {}}
    enriched = dashboard.enrich_day(day)

    assert "guide" in enriched
    assert enriched["guide"]["tasks"]["audit"]["summary"] == "Check files"
    assert enriched["guide"]["total_time_minutes"] == 20


def test_enrich_day_no_guide_field_when_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    day = {"num": 1, "week": 1, "tasks": {"audit": False}, "artifacts": {}}
    enriched = dashboard.enrich_day(day)

    assert "guide" not in enriched


def test_render_dashboard_has_clear_fallback_when_static_index_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    static_dir = tmp_path / "static"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)
    monkeypatch.setattr(dashboard, "STATIC_DIR", static_dir)
    monkeypatch.setattr(dashboard, "STATIC_INDEX", static_dir / "index.html")

    html = dashboard.render_dashboard()

    assert "React dashboard has not been built" in html
    assert "npm run build" in html
