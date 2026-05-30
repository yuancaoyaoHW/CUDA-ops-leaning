import { render, screen } from "@testing-library/react";
import { CurrentFocusPanel } from "./CurrentFocusPanel";
import { ProgressOverview } from "./ProgressOverview";
import type { DashboardDay, DashboardData } from "@/types";

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

test("current focus panel shows day details and edit action", () => {
  render(<CurrentFocusPanel day={day} onEditDay={() => undefined} />);

  expect(screen.getByRole("heading", { name: day.title })).toBeInTheDocument();
  expect(screen.getByText("Next Fix")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /edit day 01/i })).toBeInTheDocument();
  expect(screen.getByText("0/2")).toBeInTheDocument();
  expect(screen.getByText("0/1")).toBeInTheDocument();
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
