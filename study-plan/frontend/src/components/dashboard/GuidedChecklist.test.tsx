import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

test("renders task summary and time badge in collapsed state", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  expect(screen.getByText("审计仓库中已有的 kernel 实现")).toBeInTheDocument();
  expect(screen.getByText("30m")).toBeInTheDocument();
});

test("renders steps after expanding", async () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  // Steps hidden by default
  expect(screen.queryByText("列出 kernels/ 目录")).not.toBeInTheDocument();

  // Click to expand the audit item
  await userEvent.click(screen.getByRole("button", { name: /Audit/i }));

  expect(screen.getByText("列出 kernels/ 目录")).toBeInTheDocument();
  expect(screen.getByText("检查 test 覆盖")).toBeInTheDocument();
});

test("renders done_when after expanding", async () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  await userEvent.click(screen.getByRole("button", { name: /Audit/i }));

  expect(screen.getByText("docs/audit.md 存在")).toBeInTheDocument();
});

test("renders local refs as plain text, external refs as links after expanding", async () => {
  const guidesWithExternal: Record<string, TaskGuide> = {
    audit: {
      ...taskGuides.audit,
      refs: [
        { title: "Kernels 目录", url: "./kernels/" },
        { title: "Nsight Docs", url: "https://docs.nvidia.com/nsight-compute/" },
      ],
    },
  };
  render(
    <GuidedChecklist title="Tasks" items={{ audit: false }} guides={guidesWithExternal} />,
  );

  await userEvent.click(screen.getByRole("button", { name: /Audit/i }));

  // Local ref is NOT a link
  expect(screen.queryByRole("link", { name: "Kernels 目录" })).not.toBeInTheDocument();
  expect(screen.getByText("Kernels 目录")).toBeInTheDocument();

  // External ref IS a link
  const extLink = screen.getByRole("link", { name: "Nsight Docs" });
  expect(extLink).toHaveAttribute("href", "https://docs.nvidia.com/nsight-compute/");
});

test("shows blocked badge when prerequisite not done", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  expect(screen.getByText("blocked")).toBeInTheDocument();
});

test("shows dependency detail after expanding blocked item", async () => {
  render(<GuidedChecklist title="Tasks" items={tasks} guides={taskGuides} />);

  await userEvent.click(screen.getByRole("button", { name: /Plan/i }));

  expect(screen.getByText(/需要先完成: Audit/)).toBeInTheDocument();
});

test("no blocked badge when prerequisite is done", () => {
  render(
    <GuidedChecklist
      title="Tasks"
      items={{ audit: true, plan: false }}
      guides={taskGuides}
    />,
  );

  expect(screen.queryByText("blocked")).not.toBeInTheDocument();
});

test("falls back to simple list when no guides provided", () => {
  render(<GuidedChecklist title="Tasks" items={tasks} />);

  expect(screen.getByText("Audit")).toBeInTheDocument();
  expect(screen.queryByText("30m")).not.toBeInTheDocument();
});
