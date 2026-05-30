#!/usr/bin/env python3
"""
生成可视化前端 HTML 并启动本地服务器查看。

用法:
    python3 dashboard.py          # 生成 dashboard.html 并在浏览器打开
    python3 dashboard.py --serve  # 启动本地服务器（端口 8765）
    python3 dashboard.py --build  # 只生成 HTML，不打开
"""

import json
import sys
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading

try:
    import yaml
except ImportError:
    print("需要安装 PyYAML: pip install pyyaml")
    sys.exit(1)

PROGRESS_FILE = Path(__file__).parent / "progress.yaml"
OUTPUT_HTML = Path(__file__).parent / "dashboard.html"

DAY_TITLES = {
    1: "Parallel Reduction",
    2: "Online Softmax",
    3: "Tiled GEMM V1",
    4: "GEMM V2 + Double Buffering",
    5: "Fused RMSNorm",
    6: "Triton GEMM + Roofline",
    7: "周复习 + 周检",
    8: "FlashAttention 数学",
    9: "FlashAttention Triton",
    10: "vLLM Scheduler",
    11: "Continuous Batching",
    12: "Speculative Decoding",
    13: "量化 INT4/FP8",
    14: "周复习 + 阶段检",
    15: "SGLang RadixAttention",
    16: "PD 分离",
    17: "GQA/MQA/MLA",
    18: "项目选型",
    19: "项目开发: Scheduler",
    20: "项目开发: Model",
    21: "周复习 + 周检",
    22: "Paged Attention 实现",
    23: "Continuous Batching 实现",
    24: "开源 PR: 调研",
    25: "开源 PR: 实现",
    26: "项目完善",
    27: "项目 README",
    28: "周复习 + 阶段检",
    29: "Benchmark 完善",
    30: "Nsight 深入分析",
    31: "项目优化",
    32: "系统设计 #1-2",
    33: "系统设计 #3",
    34: "技术博客",
    35: "周复习 + 周检",
    36: "系统设计专项",
    37: "论文串讲",
    38: "模型压缩部署",
    39: "Mock Interview #1",
    40: "论文串讲 #2",
    41: "Mock Interview #2",
    42: "周复习 + 阶段检",
    43: "Mock #3 + 项目包装",
    44: "Mock #4 + 补课",
    45: "Mock #5 + Demo",
    46: "Mock #6 + 简历",
    47: "Mock #7 + 博客",
    48: "系统设计全天",
    49: "周检 + 阶段检",
    50: "投递 + 字节准备",
    51: "Mock 字节风格",
    52: "腾讯准备",
    53: "小红书准备",
    54: "美团准备",
    55: "薄弱点复习",
    56: "最终 Mock",
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


def load_progress():
    with open(PROGRESS_FILE) as f:
        return yaml.safe_load(f)


def transform_data(raw):
    """把 YAML 数据转成前端需要的 JSON 结构"""
    weeks = []
    daily_checks = []
    weekly_checks = []
    stage_checks = []

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
            tasks_done = sum(1 for v in tasks.values() if v)
            tasks_total = len(tasks)

            day_info = {
                "num": day_num,
                "title": DAY_TITLES.get(day_num, f"Day {day_num}"),
                "date": d.get("date", ""),
                "status": d.get("status", "not_started"),
                "daily_check": d.get("daily_check", 0),
                "tasks_done": tasks_done,
                "tasks_total": tasks_total,
                "notes": d.get("notes", ""),
            }
            days.append(day_info)

            if d.get("daily_check", 0) > 0:
                daily_checks.append({"day": day_num, "score": d["daily_check"]})

            if d.get("weekly_check_score"):
                weekly_checks.append({"day": day_num, "week": week_num, "score": d["weekly_check_score"]})

            if d.get("stage_check_score"):
                stage_checks.append({"day": day_num, "week": week_num, "score": d["stage_check_score"]})

        weeks.append({"num": week_num, "days": days})

    return {
        "weeks": weeks,
        "daily_checks": daily_checks,
        "weekly_checks": weekly_checks,
        "stage_checks": stage_checks,
        "milestones": MILESTONES,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Infra 学习进度</title>
<style>
:root {
  --bg: #1a1b26;
  --surface: #24283b;
  --surface2: #2f3549;
  --text: #c0caf5;
  --text-dim: #565f89;
  --accent: #7aa2f7;
  --green: #9ece6a;
  --yellow: #e0af68;
  --red: #f7768e;
  --purple: #bb9af7;
  --cyan: #7dcfff;
  --orange: #ff9e64;
  --radius: 8px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
  background: var(--bg);
  color: var(--text);
  padding: 24px;
  min-height: 100vh;
}

.container { max-width: 1200px; margin: 0 auto; }

h1 {
  font-size: 1.5rem;
  margin-bottom: 8px;
  color: var(--accent);
}

.subtitle {
  color: var(--text-dim);
  font-size: 0.85rem;
  margin-bottom: 32px;
}

/* ─── 总览卡片 ─── */
.overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}

.stat-card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px;
  text-align: center;
}

.stat-card .value {
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 4px;
}

.stat-card .label {
  font-size: 0.75rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* ─── 进度条 ─── */
.progress-section { margin-bottom: 32px; }

.progress-section h2 {
  font-size: 1rem;
  margin-bottom: 16px;
  color: var(--text);
}

.progress-bar-container {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 12px;
}

.progress-bar-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 0.8rem;
}

.progress-bar-track {
  height: 8px;
  background: var(--surface2);
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

/* ─── 周网格 ─── */
.weeks-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}

.week-card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 16px;
}

.week-card h3 {
  font-size: 0.9rem;
  margin-bottom: 12px;
  color: var(--accent);
}

.day-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid var(--surface2);
  font-size: 0.75rem;
}

.day-row:last-child { border-bottom: none; }

.day-status {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.day-status.done { background: var(--green); }
.day-status.in_progress { background: var(--yellow); }
.day-status.skipped { background: var(--red); }
.day-status.not_started { background: var(--surface2); border: 1px solid var(--text-dim); }

.day-title { flex: 1; color: var(--text); }
.day-title.dim { color: var(--text-dim); }

.day-tasks {
  font-size: 0.7rem;
  color: var(--text-dim);
}

.day-check {
  display: flex;
  gap: 2px;
}

.check-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--surface2);
}

.check-dot.filled-3 { background: var(--green); }
.check-dot.filled-2 { background: var(--yellow); }
.check-dot.filled-1 { background: var(--red); }

/* ─── 日检趋势图 ─── */
.chart-section {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 32px;
}

.chart-section h2 {
  font-size: 1rem;
  margin-bottom: 16px;
}

.chart-bars {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 80px;
  padding-top: 8px;
}

.chart-bar {
  flex: 1;
  min-width: 12px;
  max-width: 24px;
  border-radius: 3px 3px 0 0;
  position: relative;
  cursor: pointer;
  transition: opacity 0.2s;
}

.chart-bar:hover { opacity: 0.8; }

.chart-bar .tooltip {
  display: none;
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--surface2);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.7rem;
  white-space: nowrap;
  margin-bottom: 4px;
}

.chart-bar:hover .tooltip { display: block; }

.chart-labels {
  display: flex;
  gap: 4px;
  margin-top: 4px;
}

.chart-labels span {
  flex: 1;
  min-width: 12px;
  max-width: 24px;
  text-align: center;
  font-size: 0.6rem;
  color: var(--text-dim);
}

/* ─── 里程碑 ─── */
.milestones {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 32px;
}

.milestones h2 {
  font-size: 1rem;
  margin-bottom: 16px;
}

.milestone-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--surface2);
  font-size: 0.8rem;
}

.milestone-item:last-child { border-bottom: none; }

.milestone-icon {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  flex-shrink: 0;
}

.milestone-icon.done { background: var(--green); color: var(--bg); }
.milestone-icon.pending { background: var(--surface2); color: var(--text-dim); }

.milestone-week {
  color: var(--text-dim);
  font-size: 0.7rem;
  min-width: 40px;
}

/* ─── 热力图 ─── */
.heatmap {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 32px;
}

.heatmap h2 {
  font-size: 1rem;
  margin-bottom: 16px;
}

.heatmap-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 4px;
  max-width: 400px;
}

.heatmap-cell {
  aspect-ratio: 1;
  border-radius: 3px;
  position: relative;
  cursor: pointer;
  min-height: 20px;
}

.heatmap-cell .tooltip {
  display: none;
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--surface2);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.65rem;
  white-space: nowrap;
  margin-bottom: 4px;
  z-index: 10;
}

.heatmap-cell:hover .tooltip { display: block; }

.heat-0 { background: var(--surface2); }
.heat-1 { background: #2d4a3e; }
.heat-2 { background: #3d6b4f; }
.heat-3 { background: var(--green); }

.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 12px;
  font-size: 0.7rem;
  color: var(--text-dim);
}

.heatmap-legend .cell {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

/* ─── 响应式 ─── */
@media (max-width: 768px) {
  body { padding: 12px; }
  .overview { grid-template-columns: repeat(2, 1fr); }
  .weeks-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="container">
  <h1>🚀 AI Infra 8 周学习计划</h1>
  <p class="subtitle">目标：2 个月内拿到大模型推理优化岗位 offer</p>

  <!-- 总览 -->
  <div class="overview" id="overview"></div>

  <!-- 总进度条 -->
  <div class="progress-section" id="progress-bars"></div>

  <!-- 热力图 -->
  <div class="heatmap" id="heatmap-section"></div>

  <!-- 日检趋势 -->
  <div class="chart-section" id="daily-chart"></div>

  <!-- 里程碑 -->
  <div class="milestones" id="milestones"></div>

  <!-- 每周详情 -->
  <div class="weeks-grid" id="weeks-grid"></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

function render() {
  renderOverview();
  renderProgressBars();
  renderHeatmap();
  renderDailyChart();
  renderMilestones();
  renderWeeks();
}

function renderOverview() {
  const el = document.getElementById('overview');
  const allDays = DATA.weeks.flatMap(w => w.days);
  const done = allDays.filter(d => d.status === 'done').length;
  const total = allDays.length;
  const tasksD = allDays.reduce((s, d) => s + d.tasks_done, 0);
  const tasksT = allDays.reduce((s, d) => s + d.tasks_total, 0);
  const avgCheck = DATA.daily_checks.length > 0
    ? (DATA.daily_checks.reduce((s, d) => s + d.score, 0) / DATA.daily_checks.length).toFixed(1)
    : '—';
  const lastStage = DATA.stage_checks.length > 0
    ? DATA.stage_checks[DATA.stage_checks.length - 1].score
    : '—';

  el.innerHTML = `
    <div class="stat-card">
      <div class="value" style="color:var(--green)">${done}/${total}</div>
      <div class="label">天数完成</div>
    </div>
    <div class="stat-card">
      <div class="value" style="color:var(--accent)">${tasksD}/${tasksT}</div>
      <div class="label">子任务完成</div>
    </div>
    <div class="stat-card">
      <div class="value" style="color:var(--yellow)">${avgCheck}</div>
      <div class="label">日检平均分 /3</div>
    </div>
    <div class="stat-card">
      <div class="value" style="color:var(--purple)">${lastStage}</div>
      <div class="label">最近阶段检 /100</div>
    </div>
  `;
}

function renderProgressBars() {
  const el = document.getElementById('progress-bars');
  const allDays = DATA.weeks.flatMap(w => w.days);
  const done = allDays.filter(d => d.status === 'done').length;
  const total = allDays.length;
  const pct = total > 0 ? (done / total * 100) : 0;

  let weekBars = '';
  DATA.weeks.forEach(w => {
    const wd = w.days.filter(d => d.status === 'done').length;
    const wt = w.days.length;
    const wp = wt > 0 ? (wd / wt * 100) : 0;
    const color = wp >= 100 ? 'var(--green)' : wp > 0 ? 'var(--accent)' : 'var(--surface2)';
    weekBars += `
      <div class="progress-bar-container">
        <div class="progress-bar-label">
          <span>Week ${w.num}</span>
          <span>${wd}/${wt}</span>
        </div>
        <div class="progress-bar-track">
          <div class="progress-bar-fill" style="width:${wp}%;background:${color}"></div>
        </div>
      </div>
    `;
  });

  el.innerHTML = `
    <h2>📊 进度</h2>
    <div class="progress-bar-container">
      <div class="progress-bar-label">
        <span>总体进度</span>
        <span>${pct.toFixed(1)}%</span>
      </div>
      <div class="progress-bar-track">
        <div class="progress-bar-fill" style="width:${pct}%;background:var(--green)"></div>
      </div>
    </div>
    ${weekBars}
  `;
}

function renderHeatmap() {
  const el = document.getElementById('heatmap-section');
  const allDays = DATA.weeks.flatMap(w => w.days);

  let cells = '';
  allDays.forEach(d => {
    let heat = 0;
    if (d.status === 'done') heat = d.daily_check || 1;
    else if (d.status === 'in_progress') heat = 1;
    const title = `Day ${d.num}: ${d.title}`;
    cells += `<div class="heatmap-cell heat-${Math.min(heat, 3)}"><span class="tooltip">${title} (${d.status})</span></div>`;
  });

  el.innerHTML = `
    <h2>🗓️ 完成热力图</h2>
    <div class="heatmap-grid">${cells}</div>
    <div class="heatmap-legend">
      <span>少</span>
      <div class="cell heat-0"></div>
      <div class="cell heat-1"></div>
      <div class="cell heat-2"></div>
      <div class="cell heat-3"></div>
      <span>多</span>
    </div>
  `;
}

function renderDailyChart() {
  const el = document.getElementById('daily-chart');
  if (DATA.daily_checks.length === 0) {
    el.innerHTML = '<h2>📈 日检分数趋势</h2><p style="color:var(--text-dim);font-size:0.8rem;">还没有日检数据</p>';
    return;
  }

  const maxScore = 3;
  let bars = '';
  let labels = '';
  DATA.daily_checks.forEach(d => {
    const h = (d.score / maxScore) * 100;
    const color = d.score >= 3 ? 'var(--green)' : d.score >= 2 ? 'var(--yellow)' : 'var(--red)';
    bars += `<div class="chart-bar" style="height:${h}%;background:${color}"><span class="tooltip">Day ${d.day}: ${d.score}/3</span></div>`;
    labels += `<span>${d.day}</span>`;
  });

  el.innerHTML = `
    <h2>📈 日检分数趋势</h2>
    <div class="chart-bars">${bars}</div>
    <div class="chart-labels">${labels}</div>
  `;
}

function renderMilestones() {
  const el = document.getElementById('milestones');
  const allDays = DATA.weeks.flatMap(w => w.days);

  let items = '';
  DATA.milestones.forEach((m, i) => {
    const day = allDays.find(d => d.num === m.day);
    const done = day && day.status === 'done';
    const icon = done ? '✓' : (i + 1);
    const cls = done ? 'done' : 'pending';
    const week = Math.ceil(m.day / 7);
    items += `
      <div class="milestone-item">
        <div class="milestone-icon ${cls}">${icon}</div>
        <span class="milestone-week">W${week}</span>
        <span>${m.text}</span>
      </div>
    `;
  });

  el.innerHTML = `<h2>🏁 里程碑</h2>${items}`;
}

function renderWeeks() {
  const el = document.getElementById('weeks-grid');
  let html = '';

  DATA.weeks.forEach(w => {
    let rows = '';
    w.days.forEach(d => {
      const titleCls = d.status === 'not_started' ? 'dim' : '';
      let checks = '';
      for (let i = 0; i < 3; i++) {
        const cls = i < d.daily_check ? `filled-${d.daily_check}` : '';
        checks += `<div class="check-dot ${cls}"></div>`;
      }
      const tasks = d.tasks_total > 0 ? `${d.tasks_done}/${d.tasks_total}` : '';
      rows += `
        <div class="day-row">
          <div class="day-status ${d.status}"></div>
          <span class="day-title ${titleCls}">${d.num}. ${d.title}</span>
          <span class="day-tasks">${tasks}</span>
          <div class="day-check">${checks}</div>
        </div>
      `;
    });

    html += `
      <div class="week-card">
        <h3>Week ${w.num}</h3>
        ${rows}
      </div>
    `;
  });

  el.innerHTML = html;
}

render();
</script>
</body>
</html>"""


def build_html():
    data = load_progress()
    transformed = transform_data(data)
    json_data = json.dumps(transformed, ensure_ascii=False, indent=None)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json_data)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"✓ 已生成 {OUTPUT_HTML}")
    return OUTPUT_HTML


def serve(port=8765):
    build_html()
    os.chdir(OUTPUT_HTML.parent)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # 静默

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/dashboard.html"
    print(f"🌐 服务器运行在 {url}")
    print(f"   按 Ctrl+C 停止")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.shutdown()


def main():
    args = sys.argv[1:]

    if "--serve" in args:
        serve()
    elif "--build" in args:
        build_html()
    else:
        path = build_html()
        # 尝试打开浏览器
        url = f"file://{path.resolve()}"
        print(f"🌐 打开 {url}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
