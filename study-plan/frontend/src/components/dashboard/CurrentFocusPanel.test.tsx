import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { CurrentFocusPanel } from "./CurrentFocusPanel";
import { ProgressOverview } from "./ProgressOverview";
import type { DashboardDay, DashboardData, DayGuide } from "@/types";

const day = {
  num: 1,
  week: 1,
  title: "仓库校准 + Nsight Compute WSL2 验证",
  status: "not_started",
  jd_tags: ["perf", "docs"],
  tasks: { audit_existing_kernels: false, validate_ncu_wsl2: false },
  artifacts: { docs: false },
  next_fix: "先确认 ncu 是否能采集硬件计数器",
  weaknesses: "GPU 加速库证据不足",
  task_done: 0,
  task_total: 2,
  artifact_done: 0,
  artifact_total: 1,
  completion_pct: 0,
  daily_check: 0,
} satisfies DashboardDay;

test("current focus panel shows day details and edit action", async () => {
  const onEditDay = vi.fn();
  render(<CurrentFocusPanel day={day} onEditDay={onEditDay} />);

  expect(screen.getByRole("heading", { name: day.title })).toBeInTheDocument();
  expect(screen.getByText("Next Fix")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /edit day 01/i }));
  expect(onEditDay).toHaveBeenCalledWith(day);
  expect(screen.getByText("0/2")).toBeInTheDocument();
  expect(screen.getByText("0/1")).toBeInTheDocument();
});

test("checklist uses backend truthiness and exposes item status", () => {
  render(
    <CurrentFocusPanel
      day={{ ...day, tasks: { done_task: "complete", false_string_task: "false" } }}
      onEditDay={() => undefined}
    />,
  );

  expect(screen.getByLabelText("Done Task: done")).toBeInTheDocument();
  expect(screen.getByLabelText("False String Task: not done")).toBeInTheDocument();
});

const guideData: DayGuide = {
  tasks: {
    audit_existing_kernels: {
      summary: "审计仓库中已有的 kernel 实现",
      steps: ["列出 kernels/ 目录", "检查 test 覆盖"],
      done_when: "docs/audit.md 存在",
      time_minutes: 30,
      depends_on: [],
      refs: [{ title: "Kernels 目录", url: "./kernels/" }],
    },
    validate_ncu_wsl2: {
      summary: "验证 ncu 在 WSL2 下工作",
      steps: ["运行 ncu --version"],
      done_when: "ncu 输出版本号",
      time_minutes: 40,
      depends_on: [],
    },
  },
  artifacts: {
    docs: {
      summary: "产出审计文档",
      done_when: "docs/ 下有审计记录",
      time_minutes: 15,
      depends_on: ["audit_existing_kernels"],
    },
  },
  total_time_minutes: 85,
};

test("renders guided checklist when guide is present", () => {
  const dayWithGuide: DashboardDay = { ...day, guide: guideData };
  render(<CurrentFocusPanel day={dayWithGuide} onEditDay={() => undefined} />);

  expect(screen.getByText("审计仓库中已有的 kernel 实现")).toBeInTheDocument();
  expect(screen.getByText("~30min")).toBeInTheDocument();
  expect(screen.getByText("列出 kernels/ 目录")).toBeInTheDocument();
  expect(screen.getByText("docs/audit.md 存在")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Kernels 目录" })).toBeInTheDocument();
});

test("renders total time estimate when guide is present", () => {
  const dayWithGuide: DashboardDay = { ...day, guide: guideData };
  render(<CurrentFocusPanel day={dayWithGuide} onEditDay={() => undefined} />);

  expect(screen.getByText(/85min/)).toBeInTheDocument();
});

test("renders plain checklist when no guide", () => {
  render(<CurrentFocusPanel day={day} onEditDay={() => undefined} />);

  expect(screen.queryByText("~30min")).not.toBeInTheDocument();
  expect(screen.getByText("Audit Existing Kernels")).toBeInTheDocument();
});

test("overview shows summary numbers", () => {
  const data = {
    summary: {
      done_days: 0,
      total_days: 56,
      done_tasks: 3,
      total_tasks: 190,
      done_artifacts: 1,
      total_artifacts: 92,
      average_daily_check: null,
    },
  } as DashboardData;

  render(<ProgressOverview summary={data.summary} />);

  expect(screen.getByText("0/56")).toBeInTheDocument();
  expect(screen.getByText("3/190")).toBeInTheDocument();
  expect(screen.getByText("1/92")).toBeInTheDocument();
});
