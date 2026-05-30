#!/usr/bin/env python3
"""Compatibility entrypoint for the editable study dashboard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_dashboard_module():
    dashboard_path = Path(__file__).with_name("dashboard.py")
    spec = importlib.util.spec_from_file_location("study_plan_dashboard", dashboard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    dashboard = load_dashboard_module()
    dashboard.serve(port)


if __name__ == "__main__":
    main()
