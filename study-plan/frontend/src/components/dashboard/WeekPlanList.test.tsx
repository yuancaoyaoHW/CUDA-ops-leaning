import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { WeekPlanList } from "./WeekPlanList";
import type { DashboardWeek } from "@/types";

const weeks = [
  {
    num: 1,
    days: [
      {
        num: 1,
        week: 1,
        title: "Nsight setup",
        status: "not_started",
        jd_tags: ["perf"],
        task_done: 0,
        task_total: 2,
        artifact_done: 0,
        artifact_total: 1,
        completion_pct: 0,
      },
    ],
  },
] satisfies DashboardWeek[];

test("renders week rows and edit action", async () => {
  const onEditDay = vi.fn();
  render(<WeekPlanList weeks={weeks} currentWeek={1} onEditDay={onEditDay} />);

  expect(screen.getByText("Week 1")).toBeInTheDocument();
  expect(screen.getByText("Nsight setup")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: /edit day 1/i }));
  expect(onEditDay).toHaveBeenCalledWith(weeks[0].days[0]);
});
