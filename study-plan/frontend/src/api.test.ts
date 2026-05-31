import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { getDashboard, saveDay, saveLibrary, saveOperator } from "./api";

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

test("getDashboard reads progress data", async () => {
  fetchMock.mockResolvedValueOnce({
    ok: true,
    json: async () => ({ meta: { title: "Plan" }, weeks: [] }),
  });

  const data = await getDashboard();

  expect(fetchMock).toHaveBeenCalledWith("/api/progress");
  expect(data.meta.title).toBe("Plan");
});

test("saveDay posts day updates", async () => {
  fetchMock.mockResolvedValueOnce({ ok: true, json: async () => ({ ok: true }) });

  await saveDay(1, { status: "in_progress", daily_check: 2 });

  expect(fetchMock).toHaveBeenCalledWith("/api/day", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      day: 1,
      updates: { status: "in_progress", daily_check: 2, auto_status: true },
    }),
  });
});

test("saveOperator posts operator updates", async () => {
  fetchMock.mockResolvedValueOnce({ ok: true, json: async () => ({ ok: true }) });

  await saveOperator("row_softmax", { status: "benchmark_stage" });

  expect(fetchMock).toHaveBeenCalledWith("/api/operator", expect.objectContaining({
    method: "POST",
    body: JSON.stringify({
      operator: "row_softmax",
      updates: { status: "benchmark_stage" },
    }),
  }));
});

test("saveLibrary posts library updates", async () => {
  fetchMock.mockResolvedValueOnce({ ok: true, json: async () => ({ ok: true }) });

  await saveLibrary("cutlass", { status: "in_progress", evidence: ["note.md"] });

  expect(fetchMock).toHaveBeenCalledWith("/api/library", expect.objectContaining({
    method: "POST",
    body: JSON.stringify({
      library: "cutlass",
      updates: { status: "in_progress", evidence: ["note.md"] },
    }),
  }));
});

test("throws a readable error when save fails", async () => {
  fetchMock.mockResolvedValueOnce({
    ok: false,
    json: async () => ({ ok: false, error: "bad day" }),
  });

  await expect(saveDay(99, { status: "done" })).rejects.toThrow("bad day");
});
