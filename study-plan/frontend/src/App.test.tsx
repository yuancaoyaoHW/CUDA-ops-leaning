import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { App } from "./App";

const fetchMock = vi.fn();

const dashboardPayload = {
  meta: { title: "大模型推理框架/加速 8 周计划" },
  weeks: [],
  operators: {},
  gpu_libraries: {},
  tag_coverage: {},
  operator_maturity: {},
  status_counts: {},
  summary: {
    done_days: 0,
    total_days: 56,
    done_tasks: 0,
    total_tasks: 190,
    done_artifacts: 0,
    total_artifacts: 92,
    average_daily_check: null,
  },
  current_day: null,
  risks: [],
  options: {
    day_statuses: ["not_started", "in_progress", "done", "blocked", "skipped"],
    operator_statuses: ["not_started", "correctness_stage", "benchmark_stage", "profile_stage", "complete", "blocked"],
    library_statuses: ["not_started", "in_progress", "complete", "blocked"],
    tags: ["kernel", "perf"],
  },
};

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

test("loads and renders dashboard title", async () => {
  fetchMock.mockResolvedValueOnce({ ok: true, json: async () => dashboardPayload });

  render(<App />);

  expect(screen.getByText("Loading Dashboard")).toBeInTheDocument();
  expect(await screen.findByRole("heading", { name: "大模型推理框架/加速 8 周计划" })).toBeInTheDocument();
});

test("shows an error state when dashboard data cannot load", async () => {
  fetchMock.mockResolvedValueOnce({ ok: false, status: 500, json: async () => ({}) });

  render(<App />);

  await waitFor(() => {
    expect(screen.getByText("Unable To Load Dashboard")).toBeInTheDocument();
  });
  expect(screen.getByRole("alert")).toBeInTheDocument();
  expect(screen.getByText(/local dashboard server/i)).toBeInTheDocument();
});

test("refresh keeps loaded dashboard visible when refresh fails", async () => {
  fetchMock
    .mockResolvedValueOnce({ ok: true, json: async () => dashboardPayload })
    .mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({}) });

  render(<App />);

  expect(await screen.findByRole("heading", { name: "大模型推理框架/加速 8 周计划" })).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "Refresh Data" }));

  await waitFor(() => {
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
  expect(screen.getByRole("heading", { name: "大模型推理框架/加速 8 周计划" })).toBeInTheDocument();
  expect(screen.queryByText("Unable To Load Dashboard")).not.toBeInTheDocument();
});

test("refresh can load new dashboard data", async () => {
  fetchMock
    .mockResolvedValueOnce({ ok: true, json: async () => dashboardPayload })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...dashboardPayload, meta: { title: "Updated Plan" } }),
    });

  render(<App />);

  expect(await screen.findByRole("heading", { name: "大模型推理框架/加速 8 周计划" })).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "Refresh Data" }));

  expect(await screen.findByRole("heading", { name: "Updated Plan" })).toBeInTheDocument();
});
