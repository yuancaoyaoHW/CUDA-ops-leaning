#!/usr/bin/env python3
"""Build and serve an editable study-plan dashboard."""

from __future__ import annotations

import html
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    print("需要安装 PyYAML: pip install pyyaml")
    sys.exit(1)


BASE_DIR = Path(__file__).parent
PROGRESS_FILE = BASE_DIR / "progress.yaml"
OUTPUT_HTML = BASE_DIR / "dashboard.html"
STATIC_DIR = BASE_DIR / "static"
STATIC_INDEX = STATIC_DIR / "index.html"
TAG_ORDER = ["kernel", "framework", "serving", "perf", "quant", "docs", "interview"]
STATUS_OPTIONS = ["not_started", "in_progress", "done", "blocked", "skipped"]
OPERATOR_STATUS_OPTIONS = [
    "not_started",
    "correctness_stage",
    "benchmark_stage",
    "profile_stage",
    "complete",
    "blocked",
]
LIBRARY_STATUS_OPTIONS = ["not_started", "in_progress", "complete", "blocked"]
TEXT_DAY_FIELDS = ["status", "date", "verification", "weaknesses", "next_fix", "notes"]
INT_DAY_FIELDS = ["daily_check", "weekly_check_score", "stage_check_score"]
SAFE_ORIGINS = {"http://127.0.0.1", "http://localhost"}
GUIDES_DIR = BASE_DIR / "guides"


def load_progress() -> dict[str, Any]:
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_guide(day_num: int) -> dict[str, Any] | None:
    guide_file = GUIDES_DIR / f"day{day_num:02d}.yaml"
    if not guide_file.exists():
        return None
    with open(guide_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_progress(data: dict[str, Any]) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def truthy(value: Any) -> bool:
    return value is True or value == "true" or value == "complete"


def get_day_location(day_num: int) -> tuple[str, str]:
    if day_num < 1 or day_num > 56:
        raise ValueError("day must be between 1 and 56")
    week_num = (day_num - 1) // 7 + 1
    return f"week{week_num}", f"day{day_num:02d}"


def get_all_days(data: dict[str, Any]) -> list[dict[str, Any]]:
    days: list[dict[str, Any]] = []
    for week_num in range(1, 9):
        week_data = data.get(f"week{week_num}", {})
        for day_num in range((week_num - 1) * 7 + 1, week_num * 7 + 1):
            day = dict(week_data.get(f"day{day_num:02d}", {}))
            day["num"] = day_num
            day["week"] = week_num
            days.append(day)
    return days


def pct(done: int, total: int) -> float:
    return done / total * 100 if total else 0.0


def count_truthy(values: dict[str, Any] | None) -> tuple[int, int]:
    values = values or {}
    return sum(1 for value in values.values() if truthy(value)), len(values)


def derive_day_status(day_data: dict[str, Any]) -> str:
    current = str(day_data.get("status", "not_started"))
    if current in ("blocked", "skipped"):
        return current

    task_done, task_total = count_truthy(day_data.get("tasks"))
    artifact_done, artifact_total = count_truthy(day_data.get("artifacts"))
    total = task_total + artifact_total
    done = task_done + artifact_done

    if total and done == total:
        return "done"
    if done:
        return "in_progress"
    return "not_started"


def enrich_day(day: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(day)
    task_done, task_total = count_truthy(enriched.get("tasks"))
    artifact_done, artifact_total = count_truthy(enriched.get("artifacts"))
    enriched["task_done"] = task_done
    enriched["task_total"] = task_total
    enriched["artifact_done"] = artifact_done
    enriched["artifact_total"] = artifact_total
    enriched["completion_pct"] = pct(task_done + artifact_done, task_total + artifact_total)
    guide = load_guide(enriched["num"])
    if guide:
        enriched["guide"] = guide
    return enriched


def tag_coverage(days: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    coverage = {tag: {"done": 0, "planned": 0} for tag in TAG_ORDER}
    for day in days:
        for tag in day.get("jd_tags") or []:
            if tag not in coverage:
                coverage[tag] = {"done": 0, "planned": 0}
            coverage[tag]["planned"] += 1
            if day.get("status") == "done":
                coverage[tag]["done"] += 1
    return coverage


def operator_maturity(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, info in (data.get("operators") or {}).items():
        artifacts = info.get("artifacts") or {}
        done = sum(1 for value in artifacts.values() if truthy(value))
        total = len(artifacts)
        result[name] = {
            "done": done,
            "total": total,
            "pct": pct(done, total),
            "status": info.get("status", "unknown"),
        }
    return result


def status_counts(days: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_OPTIONS}
    for day in days:
        status = day.get("status", "not_started")
        counts[status] = counts.get(status, 0) + 1
    return counts


def summary(days: list[dict[str, Any]]) -> dict[str, Any]:
    done_days = sum(1 for day in days if day.get("status") == "done")
    total_tasks = sum(day.get("task_total", 0) for day in days)
    done_tasks = sum(day.get("task_done", 0) for day in days)
    total_artifacts = sum(day.get("artifact_total", 0) for day in days)
    done_artifacts = sum(day.get("artifact_done", 0) for day in days)
    scored_days = [int(day.get("daily_check", 0)) for day in days if int(day.get("daily_check", 0))]
    latest_stage = next(
        (
            int(day["stage_check_score"])
            for day in reversed(days)
            if day.get("stage_check_score")
        ),
        None,
    )
    return {
        "done_days": done_days,
        "total_days": len(days),
        "done_tasks": done_tasks,
        "total_tasks": total_tasks,
        "done_artifacts": done_artifacts,
        "total_artifacts": total_artifacts,
        "average_daily_check": round(sum(scored_days) / len(scored_days), 2) if scored_days else None,
        "latest_stage_score": latest_stage,
    }


def current_day(days: list[dict[str, Any]]) -> dict[str, Any] | None:
    for day in days:
        if day.get("status") in ("not_started", "in_progress", "blocked"):
            return day
    return None


def risks(data: dict[str, Any]) -> list[str]:
    operator_progress = {
        name: (item["done"], item["total"])
        for name, item in operator_maturity(data).items()
    }
    libraries = {
        name: info.get("status", "unknown")
        for name, info in (data.get("gpu_libraries") or {}).items()
    }

    result: list[str] = []
    if operator_progress.get("row_softmax", (0, 1))[0] < operator_progress.get("row_softmax", (0, 1))[1]:
        result.append("row_softmax 还没有闭环；这是 attention/FlashAttention 的前置短板。")
    if libraries.get("cuda_extension") == "not_started":
        result.append("CUDA extension 未开始；C++/CUDA 工程证据不足。")
    if libraries.get("cutlass") == "not_started":
        result.append("CUTLASS 未开始；GPU 加速库覆盖不足。")
    if libraries.get("cublas_or_pytorch_gemm") == "not_started":
        result.append("GEMM/cuBLAS baseline 未开始；无法支撑 GEMM 性能分析叙事。")
    return result


def get_api_data() -> dict[str, Any]:
    raw = load_progress()
    days = [enrich_day(day) for day in get_all_days(raw)]
    weeks = []
    for week_num in range(1, 9):
        week_days = [day for day in days if day["week"] == week_num]
        weeks.append({"num": week_num, "days": week_days})

    return {
        "meta": raw.get("meta", {}),
        "weeks": weeks,
        "operators": raw.get("operators", {}),
        "gpu_libraries": raw.get("gpu_libraries", {}),
        "tag_coverage": tag_coverage(days),
        "operator_maturity": operator_maturity(raw),
        "status_counts": status_counts(days),
        "summary": summary(days),
        "current_day": current_day(days),
        "risks": risks(raw),
        "options": {
            "day_statuses": STATUS_OPTIONS,
            "operator_statuses": OPERATOR_STATUS_OPTIONS,
            "library_statuses": LIBRARY_STATUS_OPTIONS,
            "tags": TAG_ORDER,
        },
    }


def update_day(day_num: int, updates: dict[str, Any]) -> bool:
    raw = load_progress()
    week_key, day_key = get_day_location(day_num)
    if week_key not in raw or day_key not in raw[week_key]:
        return False

    day_data = raw[week_key][day_key]
    for field in TEXT_DAY_FIELDS:
        if field in updates:
            day_data[field] = str(updates[field])
    for field in INT_DAY_FIELDS:
        if field in updates and updates[field] not in ("", None):
            day_data[field] = int(updates[field])

    for group_name in ("tasks", "artifacts"):
        if group_name not in updates:
            continue
        group = day_data.get(group_name) or {}
        for key, value in updates[group_name].items():
            if key in group:
                group[key] = bool(value)
        day_data[group_name] = group

    if updates.get("auto_status", True):
        day_data["status"] = derive_day_status(day_data)

    save_progress(raw)
    return True


def update_operator(name: str, updates: dict[str, Any]) -> bool:
    raw = load_progress()
    operators = raw.get("operators") or {}
    if name not in operators:
        return False

    operator = operators[name]
    if "status" in updates:
        status = str(updates["status"])
        if status not in OPERATOR_STATUS_OPTIONS:
            raise ValueError(f"unknown operator status: {status}")
        operator["status"] = status
    if "notes" in updates:
        operator["notes"] = str(updates["notes"])
    if "artifacts" in updates:
        artifacts = operator.get("artifacts") or {}
        for key, value in updates["artifacts"].items():
            if key in artifacts:
                artifacts[key] = bool(value)
        operator["artifacts"] = artifacts

    save_progress(raw)
    return True


def update_gpu_library(name: str, updates: dict[str, Any]) -> bool:
    raw = load_progress()
    libraries = raw.get("gpu_libraries") or {}
    if name not in libraries:
        return False

    library = libraries[name]
    if "status" in updates:
        status = str(updates["status"])
        if status not in LIBRARY_STATUS_OPTIONS:
            raise ValueError(f"unknown library status: {status}")
        library["status"] = status
    if "evidence" in updates:
        evidence = updates["evidence"]
        if isinstance(evidence, str):
            library["evidence"] = [line.strip() for line in evidence.splitlines() if line.strip()]
        elif isinstance(evidence, list):
            library["evidence"] = [str(item) for item in evidence if str(item).strip()]
        else:
            raise ValueError("evidence must be a list or newline-separated string")

    save_progress(raw)
    return True


def cors_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    if origin in SAFE_ORIGINS:
        return origin
    if origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:"):
        return origin
    return None


def json_response(handler: SimpleHTTPRequestHandler, payload: Any, code: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    origin = cors_origin(handler.headers.get("Origin"))
    if origin:
        handler.send_header("Access-Control-Allow-Origin", origin)
        handler.send_header("Vary", "Origin")
    handler.end_headers()
    handler.wfile.write(body)


def file_response(handler: SimpleHTTPRequestHandler, path: Path) -> None:
    body = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/progress":
            json_response(self, get_api_data())
            return
        if parsed.path in ("/", "/dashboard.html"):
            body = render_dashboard().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith("/assets/"):
            asset = STATIC_DIR / parsed.path.lstrip("/")
            if asset.exists() and asset.is_file():
                file_response(self, asset)
                return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path in ("/api/day", "/api/update"):
                ok = update_day(int(payload["day"]), payload["updates"])
            elif parsed.path == "/api/operator":
                ok = update_operator(str(payload["operator"]), payload["updates"])
            elif parsed.path == "/api/library":
                ok = update_gpu_library(str(payload["library"]), payload["updates"])
            else:
                json_response(self, {"ok": False, "error": "not found"}, code=404)
                return
            json_response(self, {"ok": ok})
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, code=400)

    def do_OPTIONS(self) -> None:
        origin = cors_origin(self.headers.get("Origin"))
        self.send_response(200)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def render_dashboard(embed_data: bool = False) -> str:
    if STATIC_INDEX.exists():
        return STATIC_INDEX.read_text(encoding="utf-8")

    title = html.escape(load_progress().get("meta", {}).get("title", "Study Plan"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #f8fafc; color: #172033; }}
main {{ max-width: 760px; margin: 80px auto; padding: 24px; background: white; border: 1px solid #dbe3ee; border-radius: 8px; }}
code {{ background: #eef2f7; border-radius: 4px; padding: 2px 4px; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<h2>React dashboard has not been built</h2>
<p>Run <code>cd study-plan/frontend && npm install && npm run build</code>, then start <code>python study-plan/dashboard.py --serve</code>.</p>
</main>
</body>
</html>
"""


def build() -> None:
    html_text = render_dashboard()
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML}")


def serve(port: int = 8765) -> None:
    build()
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://127.0.0.1:{port}/dashboard.html"
    print(f"Serving {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
        server.server_close()


def main() -> None:
    if "--serve" in sys.argv:
        port = 8765
        for arg in sys.argv[1:]:
            if arg.isdigit():
                port = int(arg)
        serve(port)
    else:
        build()
        if "--build" not in sys.argv:
            webbrowser.open(OUTPUT_HTML.as_uri())


if __name__ == "__main__":
    main()
