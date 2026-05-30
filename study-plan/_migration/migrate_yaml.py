#!/usr/bin/env python3
"""One-shot migration for study-plan/progress.yaml schema upgrade.

Adds verify-engine fields. Idempotent — safe to re-run; will not overwrite
existing fields.

Usage:
    python study-plan/_migration/migrate_yaml.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRESS_FILE = REPO_ROOT / "study-plan" / "progress.yaml"


VERIFY_DEFAULTS = {
    "note_min_lines": 30,
    "note_required_sections": ["bottleneck", "next_experiment"],
    "bench_required_keys_default": ["gbps"],
    "bench_required_keys_gemm": ["tflops"],
    "profile_extensions": [".ncu-rep", ".nsys-rep"],
    "profile_min_size_bytes": 1024,
    "pytest_timeout_seconds": 60,
}


# operator kinds (manual mapping; review against spec §2)
OPERATOR_KINDS = {
    "vector_add": "elementwise",
    "axpy": "elementwise",
    "row_sum": "reduction",
    "row_max": "reduction",
    "row_softmax": "reduction",
    "rmsnorm": "reduction",
    "flash_attention_toy": "attention",
    "int4_dequant": "quant",
}


# day -> (operator, phase) mapping derived from inference-acceleration-plan.md.
# Phase in {reference, implementation, tests, benchmark, profile, note, review}
DAY_MAP = {
    # week 1: row_softmax
    1: (None, "review"),
    2: ("row_softmax", "reference"),
    3: ("row_softmax", "tests"),
    4: ("row_softmax", "benchmark"),
    5: ("row_softmax", "profile"),
    6: ("row_softmax", "note"),
    7: (None, "review"),
    # week 2: rmsnorm + cuda extension
    8: ("rmsnorm", "reference"),
    9: ("rmsnorm", "implementation"),
    10: ("rmsnorm", "tests"),
    11: ("rmsnorm", "benchmark"),
    12: (None, "implementation"),  # cuda extension demo
    13: (None, "tests"),
    14: (None, "review"),
    # week 3: GEMM (no formal operator in operators block; leave None)
    15: (None, "benchmark"),
    16: (None, "benchmark"),
    17: (None, "profile"),
    18: (None, "review"),
    19: (None, "note"),
    20: (None, "review"),
    21: (None, "review"),
    # week 4: attention
    22: ("flash_attention_toy", "reference"),
    23: ("flash_attention_toy", "reference"),
    24: ("flash_attention_toy", "implementation"),
    25: ("flash_attention_toy", "tests"),
    26: ("flash_attention_toy", "benchmark"),
    27: ("flash_attention_toy", "note"),
    28: (None, "review"),
    # week 5: KV cache / scheduler toy
    29: (None, "implementation"),
    30: (None, "implementation"),
    31: (None, "note"),
    32: (None, "implementation"),
    33: (None, "note"),
    34: (None, "implementation"),
    35: (None, "review"),
    # week 6: vLLM/SGLang/TensorRT-LLM docs
    36: (None, "note"),
    37: (None, "note"),
    38: (None, "note"),
    39: (None, "note"),
    40: (None, "note"),
    41: (None, "note"),
    42: (None, "note"),
    43: (None, "review"),
    # week 7: quant
    44: ("int4_dequant", "reference"),
    45: ("int4_dequant", "implementation"),
    46: ("int4_dequant", "benchmark"),
    47: ("int4_dequant", "implementation"),
    48: ("int4_dequant", "note"),
    49: (None, "review"),
    # week 8: wrap-up
    **{d: (None, "review") for d in range(50, 57)},
}


WEEKLY_CHECK_DAYS = {7, 14, 21, 28, 35, 42, 49, 56}


def migrate(data: dict) -> tuple[dict, list[str]]:
    changes: list[str] = []

    # 1. meta.verify_defaults
    meta = data.setdefault("meta", {})
    if "verify_defaults" not in meta:
        meta["verify_defaults"] = VERIFY_DEFAULTS
        changes.append("added meta.verify_defaults")

    # 2. operators[*].kind
    operators = data.setdefault("operators", {})
    for op_name, op in operators.items():
        if op is None:
            continue
        if "kind" not in op:
            op["kind"] = OPERATOR_KINDS.get(op_name, "reduction")
            changes.append(f"added operators.{op_name}.kind={op['kind']}")

    # 3. week0.day00 — warmup gate
    if "week0" not in data:
        data["week0"] = {
            "day00": {
                "title": "Warmup 算子收尾（Day 1 gate）",
                "operators": ["vector_add", "axpy", "row_sum", "row_max"],
                "phase": "review",
                "date": "",
                "status": "not_started",
                "daily_check": 0,
                "jd_tags": ["kernel", "perf"],
                "tasks": {
                    "finish_axpy_note": False,
                    "finish_row_max_note": False,
                },
                "artifacts": {},
                "verification": "",
                "weaknesses": "",
                "next_fix": "",
                "notes": "",
            }
        }
        changes.append("added week0.day00 warmup gate")

    # 4. weekN.dayNN: operator/phase + weekly hook fields
    for wnum in range(1, 9):
        wkey = f"week{wnum}"
        week = data.get(wkey) or {}
        for day_key, day in week.items():
            if day is None:
                continue
            day_num = int(day_key.replace("day", ""))
            mapping = DAY_MAP.get(day_num, (None, "review"))
            operator, phase = mapping
            if operator is not None and "operator" not in day:
                day["operator"] = operator
                changes.append(f"{wkey}.{day_key}.operator={operator}")
            if "phase" not in day:
                day["phase"] = phase
                changes.append(f"{wkey}.{day_key}.phase={phase}")
            if day_num in WEEKLY_CHECK_DAYS:
                for fld in ("star_filled", "algo_drill_done", "cpp_drill_done"):
                    if fld not in day:
                        day[fld] = False
                        changes.append(f"{wkey}.{day_key}.{fld}=false")

    # Reorder top-level keys
    data = _reorder_top(data)

    return data, changes


def _reorder_top(data: dict) -> dict:
    """Keep top-level key order: meta, operators, gpu_libraries, week0..week8."""
    order = ["meta", "operators", "gpu_libraries"] + [f"week{i}" for i in range(0, 9)]
    new = {}
    for k in order:
        if k in data:
            new[k] = data[k]
    # keep any unknown keys at the end
    for k in data:
        if k not in new:
            new[k] = data[k]
    return new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print changes but don't write yaml")
    parser.add_argument("--file", type=Path, default=PROGRESS_FILE,
                        help="progress.yaml to migrate")
    args = parser.parse_args()

    data = yaml.safe_load(args.file.read_text(encoding="utf-8"))
    new_data, changes = migrate(data)

    if not changes:
        print("no changes (already migrated)")
        return 0

    for c in changes:
        print(f"  + {c}")

    if args.dry_run:
        print(f"\n[dry-run] would write {len(changes)} changes")
        return 0

    bak = args.file.with_suffix(args.file.suffix + ".pre-migration.bak")
    bak.write_bytes(args.file.read_bytes())
    args.file.write_text(
        yaml.safe_dump(new_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nwrote {args.file} (backup: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
