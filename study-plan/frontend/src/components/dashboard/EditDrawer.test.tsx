import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { EditDrawer } from "./EditDrawer";
import type { DashboardData, DashboardDay } from "@/types";

const day = {
  num: 1,
  week: 1,
  title: "Nsight setup",
  status: "not_started",
  daily_check: 0,
  tasks: { audit: false },
  artifacts: { docs: false },
  verification: "",
  weaknesses: "",
  next_fix: "",
  notes: "",
  task_done: 0,
  task_total: 1,
  artifact_done: 0,
  artifact_total: 1,
  completion_pct: 0,
} satisfies DashboardDay;

const options = {
  day_statuses: ["not_started", "in_progress", "done", "blocked", "skipped"],
  operator_statuses: ["not_started", "correctness_stage", "benchmark_stage", "profile_stage", "complete", "blocked"],
  library_statuses: ["not_started", "in_progress", "complete", "blocked"],
  tags: [],
} satisfies DashboardData["options"];

test("saves day updates and keeps checklist values", async () => {
  const onSaveDay = vi.fn().mockResolvedValue(undefined);
  render(
    <EditDrawer
      open
      target={{ type: "day", day }}
      options={options}
      onOpenChange={() => undefined}
      onSaveDay={onSaveDay}
      onSaveOperator={vi.fn()}
      onSaveLibrary={vi.fn()}
    />,
  );

  await userEvent.click(screen.getByLabelText("audit"));
  await userEvent.type(screen.getByLabelText("Next Fix"), "collect ncu metrics");
  await userEvent.click(screen.getByRole("button", { name: "Save Day" }));

  await waitFor(() => expect(onSaveDay).toHaveBeenCalledWith(1, expect.objectContaining({
    tasks: { audit: true },
    next_fix: "collect ncu metrics",
  })));
});

test("keeps drawer content visible when save fails", async () => {
  const onSaveDay = vi.fn().mockRejectedValue(new Error("save failed"));
  render(
    <EditDrawer
      open
      target={{ type: "day", day }}
      options={options}
      onOpenChange={() => undefined}
      onSaveDay={onSaveDay}
      onSaveOperator={vi.fn()}
      onSaveLibrary={vi.fn()}
    />,
  );

  await userEvent.click(screen.getByRole("button", { name: "Save Day" }));

  expect(await screen.findByText("save failed")).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /day 01/i })).toBeInTheDocument();
});

test("saves operator updates", async () => {
  const onSaveOperator = vi.fn().mockResolvedValue(undefined);
  render(
    <EditDrawer
      open
      target={{
        type: "operator",
        name: "row_softmax",
        operator: { status: "not_started", artifacts: { reference: false }, notes: "" },
      }}
      options={options}
      onOpenChange={() => undefined}
      onSaveDay={vi.fn()}
      onSaveOperator={onSaveOperator}
      onSaveLibrary={vi.fn()}
    />,
  );

  await userEvent.click(screen.getByLabelText("reference"));
  await userEvent.click(screen.getByRole("button", { name: "Save Operator" }));

  await waitFor(() => expect(onSaveOperator).toHaveBeenCalledWith("row_softmax", expect.objectContaining({
    artifacts: { reference: true },
  })));
});

test("saves library evidence", async () => {
  const onSaveLibrary = vi.fn().mockResolvedValue(undefined);
  render(
    <EditDrawer
      open
      target={{
        type: "library",
        name: "cutlass",
        library: { status: "not_started", evidence: [] },
      }}
      options={options}
      onOpenChange={() => undefined}
      onSaveDay={vi.fn()}
      onSaveOperator={vi.fn()}
      onSaveLibrary={onSaveLibrary}
    />,
  );

  await userEvent.type(screen.getByLabelText("Evidence"), "reports/cutlass.md");
  await userEvent.click(screen.getByRole("button", { name: "Save Library" }));

  await waitFor(() => expect(onSaveLibrary).toHaveBeenCalledWith("cutlass", expect.objectContaining({
    evidence: ["reports/cutlass.md"],
  })));
});
