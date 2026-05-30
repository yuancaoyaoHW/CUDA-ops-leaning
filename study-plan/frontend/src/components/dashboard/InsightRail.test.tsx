import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { InsightRail } from "./InsightRail";
import type { DashboardData } from "@/types";

const data = {
  risks: ["row_softmax 还没有闭环"],
  operator_maturity: {
    row_softmax: { done: 1, total: 6, pct: 16.7, status: "correctness_stage" },
  },
  gpu_libraries: {
    cutlass: { status: "not_started", evidence: [] },
  },
  tag_coverage: {
    kernel: { done: 1, planned: 12 },
  },
} as unknown as DashboardData;

test("renders risks, operators, libraries, and tags", async () => {
  const onEditOperator = vi.fn();
  const onEditLibrary = vi.fn();
  render(<InsightRail data={data} onEditOperator={onEditOperator} onEditLibrary={onEditLibrary} />);

  expect(screen.getByText("row_softmax 还没有闭环")).toBeInTheDocument();
  expect(screen.getByText("row_softmax")).toBeInTheDocument();
  expect(screen.getByText("cutlass")).toBeInTheDocument();
  expect(screen.getByText("kernel")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: /edit operator row_softmax/i }));
  expect(onEditOperator).toHaveBeenCalledWith("row_softmax");

  await userEvent.click(screen.getByRole("button", { name: /edit library cutlass/i }));
  expect(onEditLibrary).toHaveBeenCalledWith("cutlass");
});
