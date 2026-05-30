#!/usr/bin/env python3
"""Daily driver for the study-plan.

Usage:
    python study-plan/run.py today
    python study-plan/run.py next
    python study-plan/run.py day 5
    python study-plan/run.py day 5 show
    python study-plan/run.py day 5 done
    python study-plan/run.py day 5 reference|impl|tests|bench|profile|note
    python study-plan/run.py week 1
    python study-plan/run.py week 1 check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import verify  # noqa: E402

PROGRESS_FILE = HERE / "progress.yaml"

PHASE_ALIASES = {"impl": "implementation", "bench": "benchmark"}


# ---- data helpers -----------------------------------------------------------


def _load_data(yaml_path: Path | None = None) -> dict[str, Any]:
    import yaml

    path = yaml_path or PROGRESS_FILE
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_data(data: dict[str, Any], yaml_path: Path | None = None) -> None:
    import yaml

    path = yaml_path or PROGRESS_FILE
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def find_day(data: dict[str, Any], day_num: int):
    """Return (week_key, day_key, day_dict) or (None, None, None)."""
    day_key = f"day{day_num:02d}"
    for w in range(0, 9):
        wk = f"week{w}"
        if wk in data and day_key in (data.get(wk) or {}):
            return wk, day_key, data[wk][day_key]
    return None, None, None


def _get_operator(day: dict[str, Any]) -> str | None:
    """Return the primary operator for a day, or None."""
    if "operator" in day:
        return day["operator"]
    ops = day.get("operators")
    if ops and len(ops) > 0:
        return ops[0]
    return None


# ---- day subcommand ---------------------------------------------------------


def cmd_day(args, data: dict[str, Any]):
    n = args.day_num
    wk, dk, day = find_day(data, n)
    if day is None:
        print(f"Error: day {n} not found in progress.yaml", file=sys.stderr)
        sys.exit(2)

    sub = args.day_sub or "show"
    sub = PHASE_ALIASES.get(sub, sub)

    if sub == "show":
        _day_show(n, wk, dk, day, data)
    elif sub == "done":
        _day_done(n, wk, dk, day, data)
    else:
        _day_phase(n, wk, dk, day, data, sub)


def _day_show(n: int, wk: str, dk: str, day: dict, data: dict):
    title = day.get("title", "")
    phase = day.get("phase", "?")
    status = day.get("status", "?")
    op = _get_operator(day)

    print(f"=== Day {n}: {title} ===")
    print(f"  phase: {phase}   status: {status}")
    if op:
        print(f"  operator: {op}")

    if not op:
        print("  (no operator — review/gate day)")
        slots = day.get("slots") or {}
        if slots:
            print("  slots:")
            for key in ("main", "depth", "output"):
                if key in slots:
                    print(f"    {key}: {slots[key]}")
        return

    # Resolve paths and show artifacts
    op_data = (data.get("operators") or {}).get(op, {})
    try:
        paths_obj = verify.resolve_paths(op, op_data, REPO)
        path_for = {
            "reference": paths_obj.impl,
            "implementation": paths_obj.impl,
            "tests": paths_obj.tests,
            "benchmark": paths_obj.bench_json,
            "profile": paths_obj.profile_globs[0] if paths_obj.profile_globs else "?",
            "note": paths_obj.note,
        }

        # Run non-strict verify to get artifact booleans
        result = verify.verify(
            data=data,
            repo_root=REPO,
            target=("operator", op),
            strict=False,
            write=False,
            skip_tests=True,
        )
        row = result.operators.get(op)
        if row:
            print("  artifacts:")
            for art in ["reference", "implementation", "tests", "benchmark", "profile", "note"]:
                ok = row.artifacts.get(art, False)
                p = path_for.get(art, "?")
                try:
                    rel = Path(p).relative_to(REPO) if isinstance(p, Path) else p
                except (ValueError, TypeError):
                    rel = p
                mark = "✓" if ok else "✗"
                print(f"    {mark} {art:<18} {rel}")
        else:
            print("  (verify returned no results for this operator)")
    except Exception as exc:
        print(f"  (verify failed: {exc})")
        # Still show what we can from the day's own artifacts dict
        day_arts = day.get("artifacts", {})
        if day_arts:
            print("  artifacts (from yaml, unverified):")
            for k, v in day_arts.items():
                mark = "✓" if v else "✗"
                print(f"    {mark} {k}")

    slots = day.get("slots") or {}
    if slots:
        print("  slots:")
        for key in ("main", "depth", "output"):
            if key in slots:
                print(f"    {key}: {slots[key]}")

    # Show suggested command
    cmd = _default_command(op, op_data, phase)
    if cmd:
        print(f"\n  suggested: {cmd}")


def _day_phase(n: int, wk: str, dk: str, day: dict, data: dict, phase: str):
    """Show info for a specific phase of the day's operator."""
    op = _get_operator(day)
    if not op:
        print(f"Day {n} has no operator; nothing to do for phase '{phase}'.")
        return

    op_data = (data.get("operators") or {}).get(op, {})
    paths_obj = verify.resolve_paths(op, op_data, REPO)

    path_attr_for_phase = {
        "reference": "impl",
        "implementation": "impl",
        "tests": "tests",
        "benchmark": "bench",
        "profile": None,  # special: use profile_globs[0]
        "note": "note",
    }

    if phase not in path_attr_for_phase:
        print(f"Unknown phase: {phase}", file=sys.stderr)
        sys.exit(2)

    attr = path_attr_for_phase[phase]
    if phase == "profile":
        full = paths_obj.profile_globs[0] if paths_obj.profile_globs else None
    else:
        full = getattr(paths_obj, attr) if attr else None

    if full:
        try:
            rel = Path(full).relative_to(REPO)
        except (ValueError, TypeError):
            rel = full
        print(f"Day {n} · {op} · {phase}")
        print(f"  path: {rel}")
        if Path(full).exists():
            print("  exists: yes")
        else:
            print("  exists: no")
    else:
        print(f"Day {n} · {op} · {phase}: no path resolved")

    cmd = _default_command(op, op_data, phase)
    if cmd:
        print(f"  command: {cmd}")


def _day_done(n: int, wk: str, dk: str, day: dict, data: dict):
    """Run strict verify for the day's operator(s) and write back."""
    op = _get_operator(day)

    # Special gate for day 0 / day 1: verify warmup operators
    if n in (0, 1):
        warmup = ["vector_add", "axpy", "row_sum", "row_max"]
        print(f"Day {n} gate: strict-verifying warmup operators...")
        result = verify.verify(
            data=data,
            repo_root=REPO,
            target=("all",),
            strict=True,
            write=False,
            skip_tests=True,
        )
        count = 0
        for w_op in warmup:
            res = result.operators.get(w_op)
            if res:
                score = sum(1 for v in res.artifacts.values() if v)
                mark = "✓" if score >= 4 else "✗"
                print(f"  {mark} {w_op}: {score}/6")
                if score >= 4:
                    count += 1
        print(f"  warmup gate: {count}/4 operators at 4+ artifacts")
        verify.print_summary(result, strict=True)
        return

    if not op:
        print(f"Day {n}: review day — nothing to strict-verify.")
        return

    print(f"Day {n} done: strict-verifying {op}...")
    result = verify.verify(
        data=data,
        repo_root=REPO,
        target=("operator", op),
        strict=True,
        write=False,
        skip_tests=True,
    )
    verify.apply_results_in_memory(data, result)
    verify.print_summary(result, strict=True)

    # Write back
    _save_data(data)
    print("progress.yaml updated.")


# ---- week subcommand --------------------------------------------------------


def cmd_week(args, data: dict[str, Any]):
    w = args.week_num
    wk = f"week{w}"
    if wk not in data:
        print(f"Error: {wk} not found in progress.yaml", file=sys.stderr)
        sys.exit(2)

    sub = args.week_sub or "show"
    if sub == "show":
        _week_show(w, wk, data)
    elif sub == "check":
        _week_check(w, wk, data)
    else:
        print(f"Unknown week subcommand: {sub}", file=sys.stderr)
        sys.exit(2)


def _week_show(w: int, wk: str, data: dict):
    week_data = data.get(wk, {})
    print(f"=== Week {w} ===")
    for dk in sorted(week_data.keys()):
        day = week_data[dk]
        if not isinstance(day, dict):
            continue
        title = day.get("title", "")
        status = day.get("status", "?")
        op = _get_operator(day) or ""
        print(f"  {dk}: {title:<40} [{status}] {op}")


def _week_check(w: int, wk: str, data: dict):
    """Run strict verify for all days in the week, check STAR + drill."""
    week_data = data.get(wk, {})

    # Section 1: strict verify per day
    print(f"--- Week {w} strict verify ---")
    for dk in sorted(week_data.keys()):
        day = week_data[dk]
        if not isinstance(day, dict):
            continue
        op = _get_operator(day)
        if not op:
            continue
        result = verify.verify(
            data=data,
            repo_root=REPO,
            target=("operator", op),
            strict=True,
            write=False,
            skip_tests=True,
        )
        verify.apply_results_in_memory(data, result)
        op_result = result.operators[op]
        passed = sum(1 for v in op_result.artifacts.values() if v)
        print(f"  {dk}  {op:<14}  {passed}/6  [{op_result.status}]")

    # Section 2: STAR check
    print(f"\n--- Week {w} STAR check ---")
    star_path = REPO / "notes" / "star-weekly.md"
    star_ok = verify.check_star_filled(star_path, w, strict=False)
    mark = "✓" if star_ok else "✗"
    print(f"  {mark} STAR filled (week {w})")

    # Section 3: drill check
    print(f"\n--- Week {w} drill check ---")
    drill_path = REPO / "notes" / "algorithm-drill.md"
    algo_ok = verify.check_drill_done(drill_path, w, kind="algo")
    cpp_ok = verify.check_drill_done(drill_path, w, kind="cpp")
    mark_a = "✓" if algo_ok else "✗"
    mark_c = "✓" if cpp_ok else "✗"
    print(f"  {mark_a} Algo drill done")
    print(f"  {mark_c} Cpp drill done")

    # Write star_filled / algo_drill_done / cpp_drill_done to the review day
    # The review day is typically day07 for week1, day14 for week2, etc.
    review_day_num = w * 7
    review_dk = f"day{review_day_num:02d}"
    if review_dk in week_data:
        week_data[review_dk]["star_filled"] = star_ok
        week_data[review_dk]["algo_drill_done"] = algo_ok
        week_data[review_dk]["cpp_drill_done"] = cpp_ok
        _save_data(data)
        print(f"\n  wrote star_filled={star_ok}, algo_drill_done={algo_ok}, cpp_drill_done={cpp_ok} to {wk}.{review_dk}")


# ---- today / next -----------------------------------------------------------


def cmd_today(args, data: dict[str, Any]):
    """Show the first not_started day (skips day 0 gate)."""
    for n in range(1, 57):
        wk, dk, day = find_day(data, n)
        if day is None:
            continue
        if day.get("status") == "not_started":
            print(f"Today: Day {n}")
            _day_show(n, wk, dk, day, data)
            return
    print("All days are started or complete!")


def cmd_next(args, data: dict[str, Any]):
    """Show the next pending day (first not_started, skips day 0 gate)."""
    for n in range(1, 57):
        wk, dk, day = find_day(data, n)
        if day is None:
            continue
        if day.get("status") == "not_started":
            title = day.get("title", "")
            print(f"next: Day {n} ({title})")
            return
    print("All days are started or complete!")


# ---- command helpers --------------------------------------------------------


def _default_command(op: str, op_data: dict, phase: str) -> str | None:
    """Return a suggested shell command for the current phase."""
    paths_obj = verify.resolve_paths(op, op_data, REPO)
    try:
        if phase in ("reference", "implementation", "impl"):
            rel = paths_obj.impl.relative_to(REPO)
            return f"# open {rel}"
        elif phase == "tests":
            rel = paths_obj.tests.relative_to(REPO)
            return f"pytest {rel} -v"
        elif phase in ("benchmark", "bench"):
            rel = paths_obj.bench.relative_to(REPO)
            return f"python {rel}"
        elif phase == "profile":
            kind = op_data.get("kind", "reduction")
            if kind in ("attention",):
                return f"bash scripts/run_nsys.sh {op}"
            return f"bash scripts/run_ncu.sh {op}"
        elif phase == "note":
            rel = paths_obj.note.relative_to(REPO)
            return f"# open {rel}"
    except (ValueError, TypeError):
        pass
    return None


# ---- argparse ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Study-plan daily driver",
    )
    sub = parser.add_subparsers(dest="command")

    # day N [show|reference|impl|tests|bench|profile|note|done]
    day_p = sub.add_parser("day", help="Show or act on a specific day")
    day_p.add_argument("day_num", type=int, help="Day number (0-56)")
    day_p.add_argument(
        "day_sub",
        nargs="?",
        default="show",
        choices=["show", "reference", "impl", "implementation",
                 "tests", "bench", "benchmark", "profile", "note", "done"],
        help="Subcommand for the day",
    )

    # week N [show|check]
    week_p = sub.add_parser("week", help="Show or check a week")
    week_p.add_argument("week_num", type=int, help="Week number (0-8)")
    week_p.add_argument(
        "week_sub",
        nargs="?",
        default="show",
        choices=["show", "check"],
        help="Subcommand for the week",
    )

    # today
    sub.add_parser("today", help="Show first not_started day")

    # next
    sub.add_parser("next", help="Show next pending day")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    data = _load_data()

    if args.command == "day":
        cmd_day(args, data)
    elif args.command == "week":
        cmd_week(args, data)
    elif args.command == "today":
        cmd_today(args, data)
    elif args.command == "next":
        cmd_next(args, data)


if __name__ == "__main__":
    main()
