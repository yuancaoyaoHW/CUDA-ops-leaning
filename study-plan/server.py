#!/usr/bin/env python3
"""
学习进度 API 服务器

用法:
    python3 server.py          # 启动服务器 http://localhost:8765
    python3 server.py 9000     # 指定端口
"""

import json
import sys
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser
import threading

try:
    import yaml
except ImportError:
    print("需要安装 PyYAML: pip install pyyaml")
    sys.exit(1)

BASE_DIR = Path(__file__).parent
PROGRESS_FILE = BASE_DIR / "progress.yaml"


def load_progress():
    with open(PROGRESS_FILE) as f:
        return yaml.safe_load(f)


def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


DAY_TITLES = {
    1: "Parallel Reduction", 2: "Online Softmax", 3: "Tiled GEMM V1",
    4: "GEMM V2 + Double Buffering", 5: "Fused RMSNorm",
    6: "Triton GEMM + Roofline", 7: "周复习 + 周检",
    8: "FlashAttention 数学", 9: "FlashAttention Triton",
    10: "vLLM Scheduler", 11: "Continuous Batching",
    12: "Speculative Decoding", 13: "量化 INT4/FP8", 14: "周复习 + 阶段检",
    15: "SGLang RadixAttention", 16: "PD 分离", 17: "GQA/MQA/MLA",
    18: "项目选型", 19: "项目开发: Scheduler", 20: "项目开发: Model",
    21: "周复习 + 周检", 22: "Paged Attention 实现",
    23: "Continuous Batching 实现", 24: "开源 PR: 调研",
    25: "开源 PR: 实现", 26: "项目完善", 27: "项目 README",
    28: "周复习 + 阶段检", 29: "Benchmark 完善", 30: "Nsight 深入分析",
    31: "项目优化", 32: "系统设计 #1-2", 33: "系统设计 #3",
    34: "技术博客", 35: "周复习 + 周检", 36: "系统设计专项",
    37: "论文串讲", 38: "模型压缩部署", 39: "Mock Interview #1",
    40: "论文串讲 #2", 41: "Mock Interview #2", 42: "周复习 + 阶段检",
    43: "Mock #3 + 项目包装", 44: "Mock #4 + 补课", 45: "Mock #5 + Demo",
    46: "Mock #6 + 简历", 47: "Mock #7 + 博客", 48: "系统设计全天",
    49: "周检 + 阶段检", 50: "投递 + 字节准备", 51: "Mock 字节风格",
    52: "腾讯准备", 53: "小红书准备", 54: "美团准备",
    55: "薄弱点复习", 56: "最终 Mock",
}

MILESTONES = [
    {"day": 7, "text": "5 个 kernel 有 benchmark；闭卷写 reduction + softmax"},
    {"day": 14, "text": "FlashAttention 通过正确性测试；vLLM 流程图完成"},
    {"day": 21, "text": "INT4 dequant kernel 有 benchmark；SGLang 对比文档"},
    {"day": 28, "text": "Mini inference engine 可运行 或 PR 已提交"},
    {"day": 35, "text": "GitHub repo README 有完整 benchmark 图表"},
    {"day": 42, "text": "Mock interview ≥ 60 分"},
    {"day": 49, "text": "Mock interview ≥ 70 分；论文 5 分钟串讲"},
    {"day": 56, "text": "Mock interview ≥ 75 分；简历定稿；开始投递"},
]


def get_api_data():
    """把 YAML 转成前端 JSON"""
    raw = load_progress()
    weeks = []
    for week_num in range(1, 9):
        week_key = f"week{week_num}"
        week_data = raw.get(week_key, {})
        if not week_data:
            continue
        days = []
        for day_key in sorted(week_data.keys(), key=lambda x: int(x.replace("day", ""))):
            d = week_data[day_key]
            day_num = int(day_key.replace("day", ""))
            tasks = d.get("tasks", {})
            days.append({
                "num": day_num,
                "title": DAY_TITLES.get(day_num, f"Day {day_num}"),
                "date": d.get("date", ""),
                "status": d.get("status", "not_started"),
                "daily_check": d.get("daily_check", 0),
                "weekly_check_score": d.get("weekly_check_score", 0),
                "stage_check_score": d.get("stage_check_score", 0),
                "tasks": tasks,
                "notes": d.get("notes", ""),
            })
        weeks.append({"num": week_num, "days": days})
    return {"weeks": weeks, "milestones": MILESTONES}


def update_day(day_num, updates):
    """更新某一天的数据"""
    raw = load_progress()
    week_num = (day_num - 1) // 7 + 1
    week_key = f"week{week_num}"
    day_key = f"day{day_num:02d}"

    if week_key not in raw or day_key not in raw[week_key]:
        return False

    day_data = raw[week_key][day_key]

    if "status" in updates:
        day_data["status"] = updates["status"]
    if "daily_check" in updates:
        day_data["daily_check"] = int(updates["daily_check"])
    if "date" in updates:
        day_data["date"] = updates["date"]
    if "notes" in updates:
        day_data["notes"] = updates["notes"]
    if "tasks" in updates:
        for k, v in updates["tasks"].items():
            if k in day_data.get("tasks", {}):
                day_data["tasks"][k] = bool(v)
    if "weekly_check_score" in updates:
        day_data["weekly_check_score"] = int(updates["weekly_check_score"])
    if "stage_check_score" in updates:
        day_data["stage_check_score"] = int(updates["stage_check_score"])

    save_progress(raw)
    return True


class APIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format, *args):
        pass  # 静默日志

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/progress":
            data = get_api_data()
            self._json_response(data)
        elif parsed.path == "/":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/update":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body)
                day_num = int(payload["day"])
                updates = payload["updates"]
                ok = update_day(day_num, updates)
                self._json_response({"ok": ok})
            except Exception as e:
                self._json_response({"ok": False, "error": str(e)}, code=400)
        else:
            self._json_response({"error": "not found"}, code=404)

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(("127.0.0.1", port), APIHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"🚀 学习进度面板运行在 {url}")
    print(f"   按 Ctrl+C 停止")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
