#!/usr/bin/env python3
"""
AI Infra 学习进度可视化工具

用法:
    python3 progress.py          # 显示总览仪表盘
    python3 progress.py week     # 显示当前周详情
    python3 progress.py update   # 交互式更新今天的进度
    python3 progress.py history  # 显示日检/周检分数趋势
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yaml
except ImportError:
    print("需要安装 PyYAML: pip install pyyaml")
    sys.exit(1)

PROGRESS_FILE = Path(__file__).parent / "progress.yaml"

# ─── 颜色定义 ───────────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"
    BG_BLUE = "\033[44m"


# ─── 数据加载 ───────────────────────────────────────────────────────────────

def load_progress():
    if not PROGRESS_FILE.exists():
        print(f"{C.RED}找不到 progress.yaml{C.RESET}")
        sys.exit(1)
    with open(PROGRESS_FILE) as f:
        return yaml.safe_load(f)


def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ─── 统计计算 ───────────────────────────────────────────────────────────────

def get_all_days(data):
    """提取所有天的数据，按顺序"""
    days = []
    for week_key in sorted(data.keys()):
        if not week_key.startswith("week"):
            continue
        week_data = data[week_key]
        if not week_data:
            continue
        for day_key in sorted(week_data.keys(), key=lambda x: int(x.replace("day", ""))):
            day_data = week_data[day_key]
            day_data["_week"] = week_key
            day_data["_day"] = day_key
            day_data["_num"] = int(day_key.replace("day", ""))
            days.append(day_data)
    return days


def count_stats(days):
    total = len(days)
    done = sum(1 for d in days if d.get("status") == "done")
    in_progress = sum(1 for d in days if d.get("status") == "in_progress")
    skipped = sum(1 for d in days if d.get("status") == "skipped")
    not_started = total - done - in_progress - skipped
    return {"total": total, "done": done, "in_progress": in_progress,
            "skipped": skipped, "not_started": not_started}


def get_task_completion(days):
    """计算子任务完成率"""
    total_tasks = 0
    done_tasks = 0
    for d in days:
        tasks = d.get("tasks", {})
        for v in tasks.values():
            total_tasks += 1
            if v:
                done_tasks += 1
    return done_tasks, total_tasks


# ─── 可视化组件 ──────────────────────────────────────────────────────────────

def progress_bar(current, total, width=40, label="", color=C.GREEN):
    if total == 0:
        pct = 0
    else:
        pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    pct_str = f"{pct*100:.1f}%"
    print(f"  {label:<20} {color}{bar}{C.RESET} {pct_str} ({current}/{total})")


def status_icon(status):
    icons = {
        "done": f"{C.GREEN}✓{C.RESET}",
        "in_progress": f"{C.YELLOW}◐{C.RESET}",
        "skipped": f"{C.RED}✗{C.RESET}",
        "not_started": f"{C.DIM}○{C.RESET}",
    }
    return icons.get(status, "?")


def daily_check_bar(score):
    """日检分数可视化 (0-3)"""
    if score == 0:
        return f"{C.DIM}○○○{C.RESET}"
    elif score == 1:
        return f"{C.RED}●{C.RESET}{C.DIM}○○{C.RESET}"
    elif score == 2:
        return f"{C.YELLOW}●●{C.RESET}{C.DIM}○{C.RESET}"
    else:
        return f"{C.GREEN}●●●{C.RESET}"


# ─── 主视图 ──────────────────────────────────────────────────────────────────

def show_dashboard(data):
    days = get_all_days(data)
    stats = count_stats(days)
    tasks_done, tasks_total = get_task_completion(days)

    print()
    print(f"  {C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗{C.RESET}")
    print(f"  {C.BOLD}{C.CYAN}║       AI Infra 8 周学习计划 - 进度仪表盘            ║{C.RESET}")
    print(f"  {C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════╝{C.RESET}")
    print()

    # 总体进度
    print(f"  {C.BOLD}📊 总体进度{C.RESET}")
    progress_bar(stats["done"], stats["total"], label="天数完成")
    progress_bar(tasks_done, tasks_total, label="子任务完成", color=C.BLUE)
    print()

    # 每周进度条
    print(f"  {C.BOLD}📅 每周进度{C.RESET}")
    for week_num in range(1, 9):
        week_key = f"week{week_num}"
        week_data = data.get(week_key, {})
        if not week_data:
            continue
        week_days = [week_data[k] for k in sorted(week_data.keys(), key=lambda x: int(x.replace("day", "")))]
        done = sum(1 for d in week_days if d.get("status") == "done")
        total = len(week_days)

        # 每天的状态图标
        day_icons = " ".join(status_icon(d.get("status", "not_started")) for d in week_days)
        pct = done / total * 100 if total > 0 else 0

        # 周检分数
        last_day = week_days[-1] if week_days else {}
        weekly_score = last_day.get("weekly_check_score", "")
        weekly_str = f" 周检:{weekly_score}/21" if weekly_score else ""

        print(f"  Week {week_num}: {day_icons}  {C.DIM}{pct:.0f}%{C.RESET}{weekly_str}")

    print()

    # 日检趋势
    daily_scores = [(d["_num"], d.get("daily_check", 0)) for d in days if d.get("daily_check", 0) > 0]
    if daily_scores:
        print(f"  {C.BOLD}📈 日检分数趋势{C.RESET}")
        print(f"  ", end="")
        for day_num, score in daily_scores[-14:]:  # 最近 14 天
            print(f"D{day_num:02d}{daily_check_bar(score)} ", end="")
        print()
        avg = sum(s for _, s in daily_scores) / len(daily_scores)
        print(f"  平均: {avg:.1f}/3")
        print()

    # 阶段检分数
    stage_scores = []
    for d in days:
        if d.get("stage_check_score"):
            stage_scores.append((d["_num"], d["stage_check_score"]))
    if stage_scores:
        print(f"  {C.BOLD}🎯 阶段检分数{C.RESET}")
        for day_num, score in stage_scores:
            color = C.GREEN if score >= 70 else C.YELLOW if score >= 60 else C.RED
            bar_len = int(score / 100 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  Day {day_num:02d}: {color}{bar}{C.RESET} {score}/100 {'✓' if score >= 70 else '✗'}")
        print()

    # 里程碑检查
    print(f"  {C.BOLD}🏁 里程碑{C.RESET}")
    milestones = [
        (7, "5 个 kernel 有 benchmark；闭卷写 reduction + softmax"),
        (14, "FlashAttention 通过正确性测试；vLLM 流程图完成"),
        (21, "INT4 dequant kernel 有 benchmark；SGLang 对比文档"),
        (28, "Mini inference engine 可运行 或 PR 已提交"),
        (35, "GitHub repo README 有完整 benchmark 图表"),
        (42, "Mock interview ≥ 60 分"),
        (49, "Mock interview ≥ 70 分；论文 5 分钟串讲"),
        (56, "Mock interview ≥ 75 分；简历定稿；开始投递"),
    ]
    for day_num, desc in milestones:
        day_data = next((d for d in days if d["_num"] == day_num), None)
        if day_data and day_data.get("status") == "done":
            icon = f"{C.GREEN}✓{C.RESET}"
        elif day_data and day_data.get("status") in ("in_progress",):
            icon = f"{C.YELLOW}◐{C.RESET}"
        else:
            # 检查是否已过期
            done_days = [d["_num"] for d in days if d.get("status") == "done"]
            if done_days and max(done_days) > day_num:
                icon = f"{C.RED}✗{C.RESET}"
            else:
                icon = f"{C.DIM}○{C.RESET}"
        print(f"  {icon} W{(day_num-1)//7+1} Day {day_num:02d}: {desc}")

    print()


def show_week_detail(data):
    """显示当前周的详细信息"""
    days = get_all_days(data)

    # 找到当前周（第一个有 in_progress 或最后一个 done 的周）
    current_week = 1
    for d in days:
        if d.get("status") in ("in_progress", "done"):
            current_week = int(d["_week"].replace("week", ""))

    week_key = f"week{current_week}"
    week_data = data.get(week_key, {})

    print()
    print(f"  {C.BOLD}{C.CYAN}Week {current_week} 详情{C.RESET}")
    print(f"  {'─' * 50}")

    day_titles = {
        1: "Parallel Reduction", 2: "Online Softmax", 3: "Tiled GEMM V1",
        4: "GEMM V2 + Double Buffering", 5: "Fused RMSNorm", 6: "Triton GEMM + Roofline",
        7: "周复习 + 周检", 8: "FlashAttention 数学", 9: "FlashAttention Triton",
        10: "vLLM Scheduler", 11: "Continuous Batching", 12: "Speculative Decoding",
        13: "量化 INT4/FP8", 14: "周复习 + 阶段检",
    }

    for day_key in sorted(week_data.keys(), key=lambda x: int(x.replace("day", ""))):
        day_data = week_data[day_key]
        day_num = int(day_key.replace("day", ""))
        title = day_titles.get(day_num, "")
        status = day_data.get("status", "not_started")
        check = day_data.get("daily_check", 0)
        date = day_data.get("date", "")
        tasks = day_data.get("tasks", {})

        icon = status_icon(status)
        task_str = " ".join(
            f"{C.GREEN}■{C.RESET}" if v else f"{C.DIM}□{C.RESET}"
            for v in tasks.values()
        )
        date_str = f" ({date})" if date else ""

        print(f"  {icon} Day {day_num:02d}: {title:<30} {task_str} {daily_check_bar(check)}{date_str}")

        # 如果有笔记，显示
        notes = day_data.get("notes", "")
        if notes:
            print(f"       {C.DIM}└─ {notes}{C.RESET}")

    print()


def interactive_update(data):
    """交互式更新今天的进度"""
    days = get_all_days(data)

    # 找到当前应该做的 day（第一个 not_started 或 in_progress）
    current = None
    for d in days:
        if d.get("status") in ("not_started", "in_progress"):
            current = d
            break

    if not current:
        print(f"  {C.GREEN}所有天都已完成！{C.RESET}")
        return

    day_num = current["_num"]
    week_key = current["_week"]
    day_key = current["_day"]

    print()
    print(f"  {C.BOLD}更新 Day {day_num} 进度{C.RESET}")
    print(f"  {'─' * 40}")

    # 更新日期
    today = datetime.now().strftime("%Y-%m-%d")
    data[week_key][day_key]["date"] = today

    # 更新子任务
    tasks = data[week_key][day_key].get("tasks", {})
    for task_name in tasks:
        label = {"morning": "上午", "afternoon": "下午", "evening": "晚上",
                 "morning_kernel": "上午(Kernel)", "afternoon_theory": "下午(理论)",
                 "evening_distributed": "晚上(分布式)", "morning_review": "上午(复习)",
                 "afternoon_mock": "下午(Mock)", "evening_prep": "晚上(预习)",
                 "evening_stage": "晚上(阶段检)"}.get(task_name, task_name)
        current_val = "✓" if tasks[task_name] else "✗"
        resp = input(f"  {label} 完成了吗？[{current_val}] (y/n/回车跳过): ").strip().lower()
        if resp == "y":
            tasks[task_name] = True
        elif resp == "n":
            tasks[task_name] = False

    # 更新日检分数
    resp = input(f"  日检分数 (0-3, 当前={data[week_key][day_key].get('daily_check', 0)}): ").strip()
    if resp in ("0", "1", "2", "3"):
        data[week_key][day_key]["daily_check"] = int(resp)

    # 更新状态
    all_done = all(tasks.values())
    if all_done:
        data[week_key][day_key]["status"] = "done"
        print(f"  {C.GREEN}✓ Day {day_num} 标记为完成！{C.RESET}")
    else:
        data[week_key][day_key]["status"] = "in_progress"
        print(f"  {C.YELLOW}◐ Day {day_num} 标记为进行中{C.RESET}")

    # 笔记
    notes = input(f"  今日笔记/卡点 (回车跳过): ").strip()
    if notes:
        data[week_key][day_key]["notes"] = notes

    # 周检/阶段检分数（如果是复习日）
    if "weekly_check_score" in data[week_key][day_key]:
        resp = input(f"  周检分数 (0-21, 回车跳过): ").strip()
        if resp:
            data[week_key][day_key]["weekly_check_score"] = int(resp)

    if "stage_check_score" in data[week_key][day_key]:
        resp = input(f"  阶段检分数 (0-100, 回车跳过): ").strip()
        if resp:
            data[week_key][day_key]["stage_check_score"] = int(resp)

    save_progress(data)
    print(f"\n  {C.GREEN}已保存！{C.RESET}")
    print()


def show_history(data):
    """显示分数趋势图"""
    days = get_all_days(data)

    daily_scores = [(d["_num"], d.get("daily_check", 0)) for d in days if d.get("status") == "done"]

    if not daily_scores:
        print(f"  {C.DIM}还没有完成的天数{C.RESET}")
        return

    print()
    print(f"  {C.BOLD}📈 日检分数历史{C.RESET}")
    print()

    # ASCII 图表
    max_score = 3
    height = 6
    width = min(len(daily_scores), 56)
    scores_to_show = daily_scores[-width:]

    for row in range(height, -1, -1):
        threshold = row / height * max_score
        line = f"  {threshold:.1f} │"
        for _, score in scores_to_show:
            if score >= threshold and score > 0:
                if score == 3:
                    line += f"{C.GREEN}█{C.RESET}"
                elif score == 2:
                    line += f"{C.YELLOW}█{C.RESET}"
                else:
                    line += f"{C.RED}█{C.RESET}"
            else:
                line += " "
        print(line)

    print(f"      └{'─' * width}")
    # X 轴标签
    label_line = "       "
    for i, (day_num, _) in enumerate(scores_to_show):
        if i % 7 == 0:
            label_line += f"D{day_num:<5}"[0]
        else:
            label_line += " "
    print(label_line)

    # 统计
    avg = sum(s for _, s in daily_scores) / len(daily_scores)
    streak = 0
    for _, s in reversed(daily_scores):
        if s >= 2:
            streak += 1
        else:
            break

    print()
    print(f"  平均分: {avg:.2f}/3")
    print(f"  连续通过: {streak} 天")
    print(f"  总完成: {len(daily_scores)}/56 天")
    print()

    # 周检趋势
    weekly_scores = []
    for d in days:
        ws = d.get("weekly_check_score")
        if ws:
            weekly_scores.append((d["_num"], ws))

    if weekly_scores:
        print(f"  {C.BOLD}📊 周检分数{C.RESET}")
        for day_num, score in weekly_scores:
            week_num = (day_num - 1) // 7 + 1
            color = C.GREEN if score >= 15 else C.YELLOW if score >= 12 else C.RED
            bar = "█" * (score) + "░" * (21 - score)
            status = "PASS" if score >= 15 else "FAIL"
            print(f"  W{week_num}: {color}{bar}{C.RESET} {score}/21 {status}")
        print()


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    data = load_progress()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "update":
        interactive_update(data)
    elif cmd == "week":
        show_week_detail(data)
    elif cmd == "history":
        show_history(data)
    else:
        show_dashboard(data)


if __name__ == "__main__":
    main()
