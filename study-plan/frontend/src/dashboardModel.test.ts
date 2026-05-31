import { describe, expect, test } from "vitest";
import { filterWeeks, formatStatus, flattenDays, truthy } from "./dashboardModel";
import type { DashboardData } from "./types";

const data = {
  weeks: [
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
        {
          num: 2,
          week: 1,
          title: "row_softmax",
          status: "done",
          jd_tags: ["kernel"],
          task_done: 3,
          task_total: 3,
          artifact_done: 1,
          artifact_total: 1,
          completion_pct: 100,
        },
      ],
    },
  ],
} as DashboardData;

describe("dashboard model helpers", () => {
  test("truthy accepts true strings and complete markers", () => {
    expect(truthy(true)).toBe(true);
    expect(truthy("true")).toBe(true);
    expect(truthy("complete")).toBe(true);
    expect(truthy(false)).toBe(false);
  });

  test("flattenDays returns all days", () => {
    expect(flattenDays(data)).toHaveLength(2);
  });

  test("filterWeeks defaults to all weeks", () => {
    expect(filterWeeks(data, { week: "all", status: "all", tag: "all", query: "" })[0].days).toHaveLength(2);
  });

  test("filterWeeks filters by tag and search", () => {
    const weeks = filterWeeks(data, { week: "all", status: "all", tag: "kernel", query: "softmax" });
    expect(weeks[0].days.map((day) => day.title)).toEqual(["row_softmax"]);
  });

  test("formatStatus maps backend status labels", () => {
    expect(formatStatus("not_started")).toBe("Not Started");
    expect(formatStatus("benchmark_stage")).toBe("Benchmark");
  });
});
