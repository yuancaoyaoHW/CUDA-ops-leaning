import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { GuidedChecklist } from "./GuidedChecklist";
import type { Checklist, TaskGuide } from "@/types";

const tasks: Checklist = { audit: false, plan: true };
const taskGuides: Record<string, TaskGuide> = {
  audit: {
    summary: "审计仓库中已有的 kernel 实现",
    steps: ["列出 kernels/ 目录", "检查 test 覆盖"],
    done_when: "docs/audit.md 存在",
    time_minutes: 30,
    depends_on: [],
    refs: [{ title: "Kernels 目录", url: "./kernels/" }],
  },
  plan: {
    summary: "制定计划",
    steps: ["写计划文档"],
    done_when: "plan.md 存在",
    time_minutes: 15,
    depends_on: ["audit"],
  },
};

test("renders task summary and time badge", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  expect(screen.getByText("审计仓库中已有的 kernel 实现")).toBeInTheDocument();
  expect(screen.getByText("~30min")).toBeInTheDocument();
});

test("renders steps as ordered list", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  expect(screen.getByText("列出 kernels/ 目录")).toBeInTheDocument();
  expect(screen.getByText("检查 test 覆盖")).toBeInTheDocument();
});

test("renders done_when criteria", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  expect(screen.getByText("docs/audit.md 存在")).toBeInTheDocument();
});

test("renders reference links", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  const link = screen.getByRole("link", { name: "Kernels 目录" });
  expect(link).toHaveAttribute("href", "./kernels/");
});

test("shows dependency warning when prerequisite not done", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  expect(screen.getByText(/需要先完成: Audit/)).toBeInTheDocument();
});

test("no dependency warning when prerequisite is done", () => {
  render(
    <GuidedChecklist
      title="Tasks"
      items={{ audit: true, plan: false }}
      guides={taskGuides}
    />,
  );

  expect(screen.queryByText(/需要先完成/)).not.toBeInTheDocument();
});

test("falls back to simple list when no guides provided", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} />);

  expect(screen.getByText("Audit")).toBeInTheDocument();
  expect(screen.queryByText("~30min")).not.toBeInTheDocument();
});
