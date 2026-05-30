#!/usr/bin/env python3
"""AI Infra inference acceleration study progress tool.

Usage:
    python study-plan/progress.py
    python study-plan/progress.py week
    python study-plan/progress.py history
    python study-plan/progress.py analyze
    python study-plan/progress.py update
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("需要安装 PyYAML: pip install pyyaml")
    sys.exit(1)


PROGRESS_FILE = Path(__file__).parent / "progress.yaml"
TAG_ORDER = ["kernel", "framework", "serving", "perf", "quant", "docs", "interview"]
ARTIFACT_ORDER = [
    "reference",
    "implementation",
    "tests",
    "benchmark",
    "profile",
    "note",
    "docs",
    "mock",
]


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def load_progress() -> dict[str, Any]:
    candidate = Path("study-plan/progress.yaml")
    path = candidate if candidate.exists() else PROGRESS_FILE
    if not path.exists():
        print(f"{C.RED}找不到 progress.yaml{C.RESET}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_progress(data: dict[str, Any]) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def truthy(value: Any) -> bool:
    return value is True or value == "true" or value == "complete"


def week_sort_key(week_key: str) -> int:
    return int(week_key.replace("week", ""))


def day_sort_key(day_key: str) -> int:
    return int(day_key.replace("day", ""))


def get_all_days(data: dict[str, Any]) -> list[dict[str, Any]]:
    days: list[dict[str, Any]] = []
    for week_key in sorted((k for k in data if k.startswith("week")), key=week_sort_key):
        week_data = data.get(week_key) or {}
        for day_key in sorted(week_data.keys(), key=day_sort_key):
            day_data = dict(week_data[day_key])
            day_data["_week"] = week_key
            day_data["_day"] = day_key
            day_data["_num"] = day_sort_key(day_key)
            days.append(day_data)
    return days


def status_icon(status: str) -> str:
    icons = {
        "done": f"{C.GREEN}✓{C.RESET}",
        "in_progress": f"{C.YELLOW}◐{C.RESET}",
        "blocked": f"{C.RED}!{C.RESET}",
        "skipped": f"{C.RED}x{C.RESET}",
        "not_started": f"{C.DIM}○{C.RESET}",
    }
    return icons.get(status, "?")


def daily_check_bar(score: int) -> str:
    if score <= 0:
        return f"{C.DIM}○○○{C.RESET}"
    if score == 1:
        return f"{C.RED}●{C.RESET}{C.DIM}○○{C.RESET}"
    if score == 2:
        return f"{C.YELLOW}●●{C.RESET}{C.DIM}○{C.RESET}"
    return f"{C.GREEN}●●●{C.RESET}"


def progress_bar(done: int, total: int, width: int = 28, color: str = C.GREEN) -> str:
    pct = done / total if total else 0
    filled = int(width * pct)
    return f"{color}{'█' * filled}{C.DIM}{'░' * (width - filled)}{C.RESET} {pct * 100:5.1f}% ({done}/{total})"


def count_stats(days: list[dict[str, Any]]) -> dict[str, int]:
    statuses = {"done": 0, "in_progress": 0, "blocked": 0, "skipped": 0, "not_started": 0}
    for day in days:
        status = day.get("status", "not_started")
        statuses[status] = statuses.get(status, 0) + 1
    statuses["total"] = len(days)
    return statuses


def task_completion(days: list[dict[str, Any]]) -> tuple[int, int]:
    done = 0
    total = 0
    for day in days:
        for value in (day.get("tasks") or {}).values():
            total += 1
            done += int(truthy(value))
    return done, total


def artifact_completion(days: list[dict[str, Any]]) -> tuple[int, int]:
    done = 0
    total = 0
    for day in days:
        for value in (day.get("artifacts") or {}).values():
            total += 1
            done += int(truthy(value))
    return done, total


def tag_coverage(days: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    coverage = {tag: {"planned": 0, "done": 0} for tag in TAG_ORDER}
    for day in days:
        for tag in day.get("jd_tags") or []:
            if tag not in coverage:
                coverage[tag] = {"planned": 0, "done": 0}
            coverage[tag]["planned"] += 1
            if day.get("status") == "done":
                coverage[tag]["done"] += 1
    return coverage


def operator_maturity(data: dict[str, Any]) -> list[tuple[str, int, int, str]]:
    result = []
    for name, info in (data.get("operators") or {}).items():
        artifacts = info.get("artifacts") or {}
        total = len(artifacts)
        done = sum(1 for value in artifacts.values() if truthy(value))
        result.append((name, done, total, info.get("status", "unknown")))
    return result


def gpu_library_coverage(data: dict[str, Any]) -> list[tuple[str, str, int]]:
    result = []
    for name, info in (data.get("gpu_libraries") or {}).items():
        evidence = info.get("evidence") or []
        result.append((name, info.get("status", "unknown"), len(evidence)))
    return result


def current_day(days: list[dict[str, Any]]) -> dict[str, Any] | None:
    for day in days:
        if day.get("status") in ("not_started", "in_progress", "blocked"):
            return day
    return None


def print_tag_coverage(days: list[dict[str, Any]]) -> None:
    print(f"  {C.BOLD}JD 标签覆盖度{C.RESET}")
    coverage = tag_coverage(days)
    for tag in TAG_ORDER:
        item = coverage.get(tag, {"planned": 0, "done": 0})
        print(f"  {tag:<10} {progress_bar(item['done'], item['planned'], color=C.BLUE)}")
    print()


def print_operator_maturity(data: dict[str, Any]) -> None:
    print(f"  {C.BOLD}算子成熟度{C.RESET}")
    for name, done, total, status in operator_maturity(data):
        color = C.GREEN if done == total else C.YELLOW if done else C.DIM
        print(f"  {name:<22} {progress_bar(done, total, color=color)}  {status}")
    print()


def print_gpu_library_coverage(data: dict[str, Any]) -> None:
    print(f"  {C.BOLD}GPU 加速库覆盖{C.RESET}")
    for name, status, evidence_count in gpu_library_coverage(data):
        if status in ("complete", "done"):
            color = C.GREEN
        elif status == "in_progress":
            color = C.YELLOW
        else:
            color = C.DIM
        print(f"  {color}{name:<24}{C.RESET} status={status:<12} evidence={evidence_count}")
    print()


def show_dashboard(data: dict[str, Any]) -> None:
    days = get_all_days(data)
    stats = count_stats(days)
    tasks_done, tasks_total = task_completion(days)
    artifacts_done, artifacts_total = artifact_completion(days)

    print()
    print(f"  {C.BOLD}{C.CYAN}大模型推理框架/加速 8 周计划{C.RESET}")
    print()
    print(f"  天数完成   {progress_bar(stats.get('done', 0), stats['total'])}")
    print(f"  子任务完成 {progress_bar(tasks_done, tasks_total, color=C.BLUE)}")
    print(f"  产物完成   {progress_bar(artifacts_done, artifacts_total, color=C.YELLOW)}")
    print()

    print(f"  {C.BOLD}每周进度{C.RESET}")
    for week_num in range(1, 9):
        week_days = [d for d in days if d["_week"] == f"week{week_num}"]
        done = sum(1 for d in week_days if d.get("status") == "done")
        icons = " ".join(status_icon(d.get("status", "not_started")) for d in week_days)
        print(f"  W{week_num}: {icons}  {progress_bar(done, len(week_days), width=12)}")
    print()

    print_tag_coverage(days)
    print_operator_maturity(data)
    print_gpu_library_coverage(data)

    next_day = current_day(days)
    if next_day:
        tags = ", ".join(next_day.get("jd_tags") or [])
        print(f"  {C.BOLD}下一步{C.RESET}")
        print(f"  Day {next_day['_num']:02d}: {next_day.get('title', '')}")
        print(f"  tags: {tags}")
        if next_day.get("next_fix"):
            print(f"  next_fix: {next_day['next_fix']}")
        print()


def show_week_detail(data: dict[str, Any]) -> None:
    days = get_all_days(data)
    current = current_day(days)
    current_week = current["_week"] if current else "week8"
    week_days = [d for d in days if d["_week"] == current_week]

    print()
    print(f"  {C.BOLD}{C.CYAN}{current_week.upper()} 详情{C.RESET}")
    print(f"  {'-' * 72}")
    for day in week_days:
        tasks = day.get("tasks") or {}
        artifacts = day.get("artifacts") or {}
        task_done = sum(1 for v in tasks.values() if truthy(v))
        artifact_done = sum(1 for v in artifacts.values() if truthy(v))
        tag_str = ",".join(day.get("jd_tags") or [])
        print(
            f"  {status_icon(day.get('status', 'not_started'))} "
            f"Day {day['_num']:02d}: {day.get('title', '')}"
        )
        print(
            f"       tasks {task_done}/{len(tasks)} | "
            f"artifacts {artifact_done}/{len(artifacts)} | "
            f"check {daily_check_bar(int(day.get('daily_check', 0)))} | {tag_str}"
        )
        if day.get("weaknesses"):
            print(f"       weakness: {day['weaknesses']}")
        if day.get("next_fix"):
            print(f"       next: {day['next_fix']}")
    print()


def show_history(data: dict[str, Any]) -> None:
    days = get_all_days(data)
    done_days = [d for d in days if d.get("status") == "done"]
    scores = [(d["_num"], int(d.get("daily_check", 0))) for d in done_days]

    if not scores:
        print(f"  {C.DIM}还没有完成的天数{C.RESET}")
        return

    print()
    print(f"  {C.BOLD}日检分数历史{C.RESET}")
    for day_num, score in scores[-28:]:
        print(f"  D{day_num:02d} {daily_check_bar(score)} {score}/3")
    avg = sum(score for _, score in scores) / len(scores)
    print(f"\n  平均分: {avg:.2f}/3")

    weekly_scores = [
        (d["_num"], d.get("weekly_check_score"))
        for d in days
        if d.get("weekly_check_score")
    ]
    if weekly_scores:
        print(f"\n  {C.BOLD}周检分数{C.RESET}")
        for day_num, score in weekly_scores:
            status = "PASS" if int(score) >= 15 else "FAIL"
            print(f"  W{(day_num - 1) // 7 + 1}: {score}/21 {status}")

    stage_scores = [
        (d["_num"], d.get("stage_check_score"))
        for d in days
        if d.get("stage_check_score")
    ]
    if stage_scores:
        print(f"\n  {C.BOLD}阶段检分数{C.RESET}")
        for day_num, score in stage_scores:
            status = "PASS" if int(score) >= 70 else "FAIL"
            print(f"  D{day_num:02d}: {score}/100 {status}")
    print()


def show_analysis(data: dict[str, Any]) -> None:
    # Refresh artifact/status fields from verify engine before rendering
    try:
        import verify
        repo_root = Path(__file__).resolve().parents[1]
        results = verify.verify(
            data=data,
            yaml_path=None,
            repo_root=repo_root,
            target=("all",),
            strict=False,
            write=False,
            skip_tests=True,
        )
        verify.apply_results_in_memory(data, results)
    except Exception as exc:
        print(f"{C.YELLOW}verify 失败，使用 yaml 静态字段: {exc}{C.RESET}")

    days = get_all_days(data)
    print()
    print(f"  {C.BOLD}{C.CYAN}进度分析{C.RESET}")
    print()
    print_tag_coverage(days)
    print_operator_maturity(data)
    print_gpu_library_coverage(data)

    risks: list[str] = []
    ops = {name: (done, total) for name, done, total, _ in operator_maturity(data)}
    libs = {name: status for name, status, _ in gpu_library_coverage(data)}

    if ops.get("row_softmax", (0, 1))[0] < ops.get("row_softmax", (0, 1))[1]:
        risks.append("row_softmax 还没有闭环；这是 attention/FlashAttention 的前置短板。")
    if libs.get("cuda_extension") == "not_started":
        risks.append("CUDA extension 未开始；腾讯/美团 JD 中 C++/CUDA 工程证据不足。")
    if libs.get("cutlass") == "not_started":
        risks.append("CUTLASS 未开始；GPU 加速库覆盖不足。")
    if libs.get("cublas_or_pytorch_gemm") == "not_started":
        risks.append("GEMM/cuBLAS baseline 未开始；无法支撑 GEMM 性能分析叙事。")

    recent_done = [d for d in days if d.get("status") == "done"]
    if len(recent_done) >= 2 and all(int(d.get("daily_check", 0)) < 2 for d in recent_done[-2:]):
        risks.append("连续两天日检低于 2 分；应暂停推进并补课。")

    print(f"  {C.BOLD}风险项{C.RESET}")
    if risks:
        for risk in risks:
            print(f"  {C.RED}- {risk}{C.RESET}")
    else:
        print(f"  {C.GREEN}- 暂无高优先级风险。{C.RESET}")
    print()

    next_day = current_day(days)
    print(f"  {C.BOLD}建议下一步{C.RESET}")
    if next_day:
        print(f"  Day {next_day['_num']:02d}: {next_day.get('title', '')}")
        print(f"  先完成 tasks 中的第一项，再记录 verification。")
    else:
        print("  56 天计划已全部完成。")
    print()


def ask_bool(prompt: str, current: Any) -> bool:
    current_label = "y" if truthy(current) else "n"
    resp = input(f"  {prompt} [{current_label}] (y/n/回车跳过): ").strip().lower()
    if resp == "y":
        return True
    if resp == "n":
        return False
    return bool(truthy(current))


def interactive_update(data: dict[str, Any]) -> None:
    days = get_all_days(data)
    current = current_day(days)
    if not current:
        print(f"  {C.GREEN}所有天都已完成。{C.RESET}")
        return

    week_key = current["_week"]
    day_key = current["_day"]
    target = data[week_key][day_key]

    print()
    print(f"  {C.BOLD}更新 Day {current['_num']:02d}: {target.get('title', '')}{C.RESET}")
    target["date"] = datetime.now().strftime("%Y-%m-%d")

    for name, value in (target.get("tasks") or {}).items():
        target["tasks"][name] = ask_bool(f"task:{name}", value)

    for name, value in (target.get("artifacts") or {}).items():
        target["artifacts"][name] = ask_bool(f"artifact:{name}", value)

    resp = input(f"  日检分数 (0-3, 当前={target.get('daily_check', 0)}): ").strip()
    if resp in ("0", "1", "2", "3"):
        target["daily_check"] = int(resp)

    if "weekly_check_score" in target:
        resp = input(f"  周检分数 (0-21, 当前={target.get('weekly_check_score', 0)}): ").strip()
        if resp:
            target["weekly_check_score"] = int(resp)

    if "stage_check_score" in target:
        resp = input(f"  阶段检分数 (0-100, 当前={target.get('stage_check_score', 0)}): ").strip()
        if resp:
            target["stage_check_score"] = int(resp)

    verification = input("  verification 命令/结果 (回车跳过): ").strip()
    if verification:
        target["verification"] = verification

    weaknesses = input("  weaknesses (回车跳过): ").strip()
    if weaknesses:
        target["weaknesses"] = weaknesses

    next_fix = input("  next_fix (回车跳过): ").strip()
    if next_fix:
        target["next_fix"] = next_fix

    notes = input("  notes (回车跳过): ").strip()
    if notes:
        target["notes"] = notes

    task_values = list((target.get("tasks") or {}).values())
    artifact_values = list((target.get("artifacts") or {}).values())
    if task_values and all(truthy(v) for v in task_values) and artifact_values and all(truthy(v) for v in artifact_values):
        target["status"] = "done"
    elif any(truthy(v) for v in task_values + artifact_values):
        target["status"] = "in_progress"
    else:
        target["status"] = "not_started"

    save_progress(data)
    print(f"\n  {C.GREEN}已保存。{C.RESET}")
    print()


def cmd_verify(args, data):
    import verify
    if args.operator:
        target = ("operator", args.operator)
    elif args.day is not None:
        target = ("day", args.day)
    else:
        target = ("all",)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        results = verify.verify(
            data=data,
            yaml_path=PROGRESS_FILE if args.write else None,
            repo_root=repo_root,
            target=target,
            strict=args.strict,
            write=args.write,
            skip_tests=args.skip_tests,
        )
    except verify.UnknownTargetError as exc:
        print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
        return 2
    except verify.MissingDependencyError as exc:
        print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
        return 4
    except verify.BackupError as exc:
        print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
        return 3
    verify.print_summary(results, strict=args.strict)
    return 0


def cmd_status(args, data):
    import verify
    if args.day is None:
        print(f"{C.RED}--day N is required{C.RESET}", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = verify.verify_day(data, args.day, repo_root=repo_root)
    except verify.UnknownTargetError as exc:
        print(f"{C.RED}{exc}{C.RESET}", file=sys.stderr)
        return 2
    verify.print_day_status(result, args.day)
    print(f"day{args.day:02d}")
    return 0


def cmd_drill(args, data):
    import verify
    repo_root = Path(__file__).resolve().parents[1]
    summary = verify.collect_drill_summary(data, repo_root=repo_root)
    verify.print_drill_summary(summary)
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="progress")
    sub = parser.add_subparsers(dest="cmd")

    p_verify = sub.add_parser("verify", help="Run verify engine")
    target_group = p_verify.add_mutually_exclusive_group()
    target_group.add_argument("--day", type=int)
    target_group.add_argument("--operator", type=str)
    target_group.add_argument("--all", action="store_true")
    p_verify.add_argument("--strict", action="store_true")
    p_verify.add_argument("--write", action="store_true")
    p_verify.add_argument("--skip-tests", action="store_true")

    p_status = sub.add_parser("status", help="Show single-day artifact status")
    p_status.add_argument("--day", type=int)

    sub.add_parser("drill", help="Show STAR/algo/cpp drill summary")
    sub.add_parser("week", help="Show current week detail")
    sub.add_parser("history", help="Show recent history")
    sub.add_parser("analyze", help="Show coverage analysis")
    sub.add_parser("update", help="Interactive update (legacy)")

    args = parser.parse_args()
    data = load_progress()

    if args.cmd == "verify":
        return cmd_verify(args, data)
    if args.cmd == "status":
        return cmd_status(args, data)
    if args.cmd == "drill":
        return cmd_drill(args, data)
    if args.cmd == "week":
        show_week_detail(data)
        return 0
    if args.cmd == "history":
        show_history(data)
        return 0
    if args.cmd == "analyze":
        show_analysis(data)
        return 0
    if args.cmd == "update":
        interactive_update(data)
        return 0

    show_dashboard(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
