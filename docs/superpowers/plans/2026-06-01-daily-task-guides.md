# Daily Task Guides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured per-task guidance (steps, done criteria, refs, time estimates, dependencies) to the study-plan dashboard, stored in per-day YAML files and rendered in the CurrentFocusPanel.

**Architecture:** Per-day YAML guide files in `study-plan/guides/` are loaded by the Python backend's `enrich_day()` and merged into the API response as a `guide` field. The React frontend reads this field and renders an enhanced Checklist with steps, completion criteria, time badges, dependency warnings, and reference links.

**Tech Stack:** Python/PyYAML (backend), React/TypeScript/Tailwind (frontend), Vitest (frontend tests), pytest (backend tests)

---

## File Structure

| Path | Role |
|------|------|
| `study-plan/guides/day01.yaml` – `day07.yaml` | Week 1 guide content (new) |
| `study-plan/dashboard.py` | Add `GUIDES_DIR`, `load_guide()`, modify `enrich_day()` |
| `tests/test_study_plan_dashboard.py` | Add guide-loading tests |
| `study-plan/frontend/src/types.ts` | Add `TaskGuide`, `ArtifactGuide`, `DayGuide` types |
| `study-plan/frontend/src/components/dashboard/GuidedChecklist.tsx` | New component (new) |
| `study-plan/frontend/src/components/dashboard/GuidedChecklist.test.tsx` | Tests for new component (new) |
| `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.tsx` | Use `GuidedChecklist` when guide exists |
| `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.test.tsx` | Add guide rendering tests |

---

## Task 1: Python Backend – Guide Loading

**Files:**
- Modify: `study-plan/dashboard.py`
- Modify: `tests/test_study_plan_dashboard.py`

- [ ] **Step 1: Write failing tests for guide loading**

Add to `tests/test_study_plan_dashboard.py`:

```python
def test_load_guide_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    result = dashboard.load_guide(1)

    assert result is None


def test_load_guide_returns_parsed_yaml(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "day01.yaml").write_text(
        "day: 1\ntasks:\n  audit:\n    summary: Check files\n    steps:\n      - list dir\n    done_when: audit.md exists\n    time_minutes: 20\n    depends_on: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    result = dashboard.load_guide(1)

    assert result is not None
    assert result["day"] == 1
    assert result["tasks"]["audit"]["summary"] == "Check files"
    assert result["tasks"]["audit"]["steps"] == ["list dir"]
    assert result["tasks"]["audit"]["time_minutes"] == 20


def test_enrich_day_merges_guide(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "day01.yaml").write_text(
        "day: 1\ntasks:\n  audit:\n    summary: Check files\n    steps:\n      - list dir\n    done_when: audit.md exists\n    time_minutes: 20\n    depends_on: []\ntotal_time_minutes: 20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    day = {"num": 1, "week": 1, "tasks": {"audit": False}, "artifacts": {}}
    enriched = dashboard.enrich_day(day)

    assert "guide" in enriched
    assert enriched["guide"]["tasks"]["audit"]["summary"] == "Check files"
    assert enriched["guide"]["total_time_minutes"] == 20


def test_enrich_day_no_guide_field_when_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    guides_dir = tmp_path / "guides"
    monkeypatch.setattr(dashboard, "GUIDES_DIR", guides_dir)

    day = {"num": 1, "week": 1, "tasks": {"audit": False}, "artifacts": {}}
    enriched = dashboard.enrich_day(day)

    assert "guide" not in enriched
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q -k "guide"
```

Expected: FAIL — `dashboard.load_guide` does not exist, `dashboard.GUIDES_DIR` does not exist.

- [ ] **Step 3: Add GUIDES_DIR constant and load_guide function**

In `study-plan/dashboard.py`, after line 42 (`SAFE_ORIGINS = ...`), add:

```python
GUIDES_DIR = BASE_DIR / "guides"
```

After the `load_progress()` function, add:

```python
def load_guide(day_num: int) -> dict[str, Any] | None:
    guide_file = GUIDES_DIR / f"day{day_num:02d}.yaml"
    if not guide_file.exists():
        return None
    with open(guide_file, encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Modify enrich_day to merge guide data**

In `study-plan/dashboard.py`, in the `enrich_day()` function, replace the final `return enriched` with:

```python
    guide = load_guide(enriched["num"])
    if guide:
        enriched["guide"] = guide
    return enriched
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q
```

Expected: all tests pass (existing 7 + 4 new guide tests = 11).

- [ ] **Step 6: Commit**

```bash
git add study-plan/dashboard.py tests/test_study_plan_dashboard.py
git commit -m "feat: add guide loading and merge into enriched day data"
```

## Task 2: Frontend Types

**Files:**
- Modify: `study-plan/frontend/src/types.ts`

- [ ] **Step 1: Add guide types to types.ts**

Insert before the `export type DayUpdates` line in `study-plan/frontend/src/types.ts`:

```typescript
export interface TaskGuide {
  summary: string;
  steps: string[];
  done_when: string;
  time_minutes: number;
  depends_on: string[];
  refs?: { title: string; url: string }[];
}

export interface ArtifactGuide {
  summary: string;
  done_when: string;
  time_minutes: number;
  depends_on: string[];
  refs?: { title: string; url: string }[];
}

export interface DayGuide {
  tasks: Record<string, TaskGuide>;
  artifacts: Record<string, ArtifactGuide>;
  total_time_minutes: number;
}
```

- [ ] **Step 2: Add guide field to DashboardDay**

In the `DashboardDay` interface, add after `completion_pct: number;`:

```typescript
  guide?: DayGuide;
```

- [ ] **Step 3: Verify build passes**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn/study-plan/frontend
npx tsc --noEmit
```

Expected: no type errors.

- [ ] **Step 4: Commit**

```bash
git add study-plan/frontend/src/types.ts
git commit -m "feat: add TaskGuide, ArtifactGuide, DayGuide types"
```

## Task 3: GuidedChecklist Component

**Files:**
- Create: `study-plan/frontend/src/components/dashboard/GuidedChecklist.tsx`
- Create: `study-plan/frontend/src/components/dashboard/GuidedChecklist.test.tsx`

- [ ] **Step 1: Write failing test for GuidedChecklist**

Create `study-plan/frontend/src/components/dashboard/GuidedChecklist.test.tsx`:

```typescript
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn/study-plan/frontend
npx vitest run src/components/dashboard/GuidedChecklist.test.tsx
```

Expected: FAIL — module `./GuidedChecklist` not found.

- [ ] **Step 3: Implement GuidedChecklist component**

Create `study-plan/frontend/src/components/dashboard/GuidedChecklist.tsx`:

```typescript
import { CheckCircle2, Clock, AlertTriangle, ExternalLink } from "lucide-react";
import { truthy } from "@/dashboardModel";
import type { Checklist as ChecklistType, TaskGuide, ArtifactGuide } from "@/types";

function humanize(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

interface GuidedChecklistProps {
  title: string;
  items: ChecklistType;
  guides?: Record<string, TaskGuide | ArtifactGuide>;
}

export function GuidedChecklist({ title, items, guides }: GuidedChecklistProps) {
  return (
    <div>
      <p className="mb-3 text-xs font-semibold text-slate-600">{title}</p>
      <ul className="grid gap-4">
        {Object.entries(items).map(([key, value]) => {
          const done = truthy(value);
          const guide = guides?.[key] as TaskGuide | undefined;

          if (!guide) {
            return (
              <li
                key={key}
                className="flex items-start gap-2 text-sm text-slate-700"
                aria-label={`${humanize(key)}: ${done ? "done" : "not done"}`}
              >
                <CheckCircle2
                  aria-hidden="true"
                  className={done ? "mt-0.5 h-4 w-4 text-emerald-600" : "mt-0.5 h-4 w-4 text-slate-300"}
                />
                <span>{humanize(key)}</span>
              </li>
            );
          }

          const blockers = (guide.depends_on || []).filter(
            (dep) => !truthy(items[dep]),
          );

          return (
            <li key={key} className="rounded-lg border border-slate-200 p-3">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2">
                  <CheckCircle2
                    aria-hidden="true"
                    className={
                      done
                        ? "mt-0.5 h-4 w-4 shrink-0 text-emerald-600"
                        : "mt-0.5 h-4 w-4 shrink-0 text-slate-300"
                    }
                  />
                  <div>
                    <span className="text-sm font-medium text-slate-900">
                      {humanize(key)}
                    </span>
                    <p className="mt-0.5 text-xs text-slate-500">
                      {guide.summary}
                    </p>
                  </div>
                </div>
                <span className="flex shrink-0 items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
                  <Clock className="h-3 w-3" aria-hidden="true" />
                  ~{guide.time_minutes}min
                </span>
              </div>

              {blockers.length > 0 ? (
                <p className="mt-2 flex items-center gap-1 text-xs text-amber-700">
                  <AlertTriangle className="h-3 w-3" aria-hidden="true" />
                  需要先完成: {blockers.map((b) => humanize(b)).join(", ")}
                </p>
              ) : null}

              {"steps" in guide && guide.steps.length > 0 ? (
                <ol className="mt-2 ml-6 list-decimal space-y-1 text-xs text-slate-700">
                  {guide.steps.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ol>
              ) : null}

              <div className="mt-2 rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs text-emerald-800">
                ✅ {guide.done_when}
              </div>

              {guide.refs && guide.refs.length > 0 ? (
                <div className="mt-2 flex flex-wrap gap-2">
                  {guide.refs.map((ref, i) => (
                    <a
                      key={i}
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-blue-600 hover:underline"
                    >
                      <ExternalLink className="h-3 w-3" aria-hidden="true" />
                      {ref.title}
                    </a>
                  ))}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn/study-plan/frontend
npx vitest run src/components/dashboard/GuidedChecklist.test.tsx
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add study-plan/frontend/src/components/dashboard/GuidedChecklist.tsx \
       study-plan/frontend/src/components/dashboard/GuidedChecklist.test.tsx
git commit -m "feat: add GuidedChecklist component with steps, time, deps, refs"
```

## Task 4: Integrate GuidedChecklist into CurrentFocusPanel

**Files:**
- Modify: `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.test.tsx`

- [ ] **Step 1: Add test for guide rendering in CurrentFocusPanel**

Add to `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.test.tsx`:

```typescript
import type { DashboardDay, DayGuide } from "@/types";

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn/study-plan/frontend
npx vitest run src/components/dashboard/CurrentFocusPanel.test.tsx
```

Expected: new tests fail (guide rendering not implemented yet).

- [ ] **Step 3: Update CurrentFocusPanel to use GuidedChecklist**

In `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.tsx`:

1. Add import at top:

```typescript
import { GuidedChecklist } from "./GuidedChecklist";
```

2. Replace the existing checklist grid section:

```typescript
        <div className="grid gap-4 md:grid-cols-2">
          <Checklist title="Tasks" items={day.tasks || {}} />
          <Checklist title="Artifacts" items={day.artifacts || {}} />
        </div>
```

With:

```typescript
        {day.guide ? (
          <>
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-slate-600">预估总时间</p>
              <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
                ~{day.guide.total_time_minutes}min
              </span>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <GuidedChecklist title="Tasks" items={day.tasks || {}} guides={day.guide.tasks} />
              <GuidedChecklist title="Artifacts" items={day.artifacts || {}} guides={day.guide.artifacts} />
            </div>
          </>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            <Checklist title="Tasks" items={day.tasks || {}} />
            <Checklist title="Artifacts" items={day.artifacts || {}} />
          </div>
        )}
```

- [ ] **Step 4: Run all frontend tests**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn/study-plan/frontend
npm test
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add study-plan/frontend/src/components/dashboard/CurrentFocusPanel.tsx \
       study-plan/frontend/src/components/dashboard/CurrentFocusPanel.test.tsx
git commit -m "feat: integrate GuidedChecklist into CurrentFocusPanel"
```

## Task 5: Week 1 Guide Content (day01–day07)

**Files:**
- Create: `study-plan/guides/day01.yaml`
- Create: `study-plan/guides/day02.yaml`
- Create: `study-plan/guides/day03.yaml`
- Create: `study-plan/guides/day04.yaml`
- Create: `study-plan/guides/day05.yaml`
- Create: `study-plan/guides/day06.yaml`
- Create: `study-plan/guides/day07.yaml`

- [ ] **Step 1: Create guides directory**

```bash
mkdir -p study-plan/guides
```

- [ ] **Step 2: Create day01.yaml**

Create `study-plan/guides/day01.yaml`:

```yaml
day: 1

tasks:
  audit_existing_kernels:
    summary: 审计仓库中已有的 kernel 实现，梳理现有代码和测试覆盖
    steps:
      - 列出 kernels/ 目录下所有 .py 和 .cu 文件
      - 对每个 kernel 检查是否有对应的 test_*.py 文件
      - 检查每个 kernel 是否有 benchmark 脚本
      - 将结果记录到 docs/kernel-audit.md（表格形式：kernel名/有无test/有无benchmark/状态）
    done_when: docs/kernel-audit.md 存在，列出所有 kernel 的测试和 benchmark 覆盖状态
    time_minutes: 30
    depends_on: []
    refs:
      - title: 项目 kernels 目录
        url: ./kernels/

  map_operator_maturity:
    summary: 梳理 operator 成熟度矩阵，对照 progress.yaml 中 operators 部分
    steps:
      - 打开 progress.yaml，找到 operators 部分
      - 对每个 operator 检查其 artifacts（reference/implementation/tests/benchmark/profile/note）
      - 对照实际文件确认 artifacts 状态是否准确
      - 更新 progress.yaml 中不准确的 operator status
    done_when: progress.yaml 中所有 operator 的 status 和 artifacts 反映真实文件状态
    time_minutes: 25
    depends_on: [audit_existing_kernels]
    refs:
      - title: progress.yaml operators 部分
        url: ./progress.yaml

  validate_ncu_wsl2:
    summary: 验证 Nsight Compute 在 WSL2 下能采集硬件计数器
    steps:
      - 运行 ncu --version 确认安装版本
      - 编译或找到一个简单 CUDA kernel（如 vectorAdd）
      - 运行 ncu --set full -o test_profile ./vectorAdd
      - 打开 .ncu-rep 文件，确认 sm__throughput.avg.pct_of_peak_sustained_elapsed 等硬件计数器有数据
      - 如果失败，记录错误信息和可能的 workaround（如需要 --target-processes all）
    done_when: ncu 能输出包含硬件计数器的 .ncu-rep 文件，或记录了明确的失败原因和 workaround
    time_minutes: 40
    depends_on: []
    refs:
      - title: Nsight Compute CLI 文档
        url: https://docs.nvidia.com/nsight-compute/NsightComputeCli/
      - title: WSL2 GPU 支持说明
        url: https://docs.nvidia.com/cuda/wsl-user-guide/
      - title: NCU WSL2 已知限制
        url: https://developer.nvidia.com/nsight-compute

artifacts:
  docs:
    summary: 产出当天的审计文档（kernel-audit.md）
    done_when: docs/kernel-audit.md 存在且内容完整
    time_minutes: 10
    depends_on: [audit_existing_kernels, map_operator_maturity]

total_time_minutes: 105
```

- [ ] **Step 3: Create day02.yaml**

Create `study-plan/guides/day02.yaml`:

```yaml
day: 2

tasks:
  write_reference:
    summary: 用 PyTorch 实现 row_softmax 参考版本，作为正确性基准
    steps:
      - 创建 kernels/softmax/reference.py
      - 实现 row_softmax(x) 函数，输入 (M, N) tensor，沿 dim=-1 做 softmax
      - 处理数值稳定性：先减去每行最大值
      - 添加 __main__ 块验证与 torch.softmax 结果一致（atol=1e-6）
    done_when: python kernels/softmax/reference.py 运行通过，输出 "PASS"
    time_minutes: 25
    depends_on: []
    refs:
      - title: PyTorch softmax 源码
        url: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SoftMax.cpp
      - title: Online Softmax 论文
        url: https://arxiv.org/abs/1805.02867

  write_triton_skeleton:
    summary: 用 Triton 实现 row_softmax kernel 骨架
    steps:
      - 创建 kernels/softmax/triton_softmax.py
      - 定义 @triton.jit kernel 函数签名（input_ptr, output_ptr, n_cols, BLOCK_SIZE）
      - 实现 row-wise 并行：每个 program 处理一行
      - 实现 load → max → subtract → exp → sum → divide → store 流程
      - 用 tl.where 处理 BLOCK_SIZE > n_cols 的 padding
      - 添加 wrapper 函数分配输出 tensor 并 launch kernel
    done_when: python kernels/softmax/triton_softmax.py 运行通过，与 reference 结果一致（atol=1e-5）
    time_minutes: 40
    depends_on: [write_reference]
    refs:
      - title: Triton softmax 教程
        url: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
      - title: Triton 编程模型
        url: https://triton-lang.org/main/programming-guide/chapter-1/introduction.html

  record_addressing:
    summary: 记录 Triton kernel 的寻址模式和 BLOCK_SIZE 选择逻辑
    steps:
      - 在 kernels/softmax/README.md 中记录
      - 画出 grid/block 映射关系（哪个 program 对应哪行）
      - 解释 BLOCK_SIZE 如何影响 shared memory 和 occupancy
      - 记录为什么用 tl.where 做 mask 而不是 if 分支
    done_when: kernels/softmax/README.md 存在且包含寻址模式说明
    time_minutes: 20
    depends_on: [write_triton_skeleton]

artifacts:
  reference:
    summary: PyTorch 参考实现文件
    done_when: kernels/softmax/reference.py 存在且可运行
    time_minutes: 0
    depends_on: [write_reference]
  implementation:
    summary: Triton kernel 实现文件
    done_when: kernels/softmax/triton_softmax.py 存在且可运行
    time_minutes: 0
    depends_on: [write_triton_skeleton]
  note:
    summary: 寻址模式笔记
    done_when: kernels/softmax/README.md 存在
    time_minutes: 0
    depends_on: [record_addressing]

total_time_minutes: 85
```

- [ ] **Step 4: Create day03.yaml**

Create `study-plan/guides/day03.yaml`:

```yaml
day: 3

tasks:
  aligned_tests:
    summary: 编写对齐尺寸的 softmax 正确性测试
    steps:
      - 创建 kernels/softmax/test_softmax.py
      - 用 pytest 参数化测试不同 (M, N) 组合：(1,128), (32,256), (64,1024), (128,4096)
      - 对每组参数比较 triton_softmax 与 torch.softmax 的输出（atol=1e-5, rtol=1e-5）
      - 测试 float32 和 float16 两种 dtype
    done_when: pytest kernels/softmax/test_softmax.py::test_aligned -v 全部 PASS
    time_minutes: 25
    depends_on: []
    refs:
      - title: pytest parametrize 文档
        url: https://docs.pytest.org/en/stable/how-to/parametrize.html

  non_aligned_tests:
    summary: 编写非对齐尺寸的 softmax 正确性测试（N 不是 2 的幂）
    steps:
      - 在 test_softmax.py 中添加 test_non_aligned
      - 测试 N=33, 127, 1000, 3999 等非 2 幂尺寸
      - 确认 tl.where mask 正确处理了 padding 区域
      - 如果失败，修复 triton kernel 中的 mask 逻辑
    done_when: pytest kernels/softmax/test_softmax.py::test_non_aligned -v 全部 PASS
    time_minutes: 20
    depends_on: [aligned_tests]

  dtype_and_mask_tests:
    summary: 编写 dtype 边界和 mask 相关测试
    steps:
      - 添加 test_large_values 测试极大值输入（不溢出）
      - 添加 test_zero_row 测试全零行
      - 添加 test_negative_row 测试全负数行
      - 添加 test_single_element 测试 N=1 的情况
      - 确认所有情况下输出行和为 1.0
    done_when: pytest kernels/softmax/test_softmax.py -v 全部 PASS（包含边界测试）
    time_minutes: 25
    depends_on: [aligned_tests]

artifacts:
  tests:
    summary: 完整的 softmax 测试文件
    done_when: kernels/softmax/test_softmax.py 存在且 pytest 全部通过
    time_minutes: 0
    depends_on: [aligned_tests, non_aligned_tests, dtype_and_mask_tests]

total_time_minutes: 70
```

- [ ] **Step 5: Create day04.yaml**

Create `study-plan/guides/day04.yaml`:

```yaml
day: 4

tasks:
  write_benchmark:
    summary: 编写 softmax benchmark 脚本，测量不同尺寸下的性能
    steps:
      - 创建 kernels/softmax/bench_softmax.py
      - 使用 triton.testing.do_bench 测量 kernel 执行时间
      - 参数化 M=[128,512,2048,8192] 和 N=[128,1024,4096,16384]
      - 输出表格：M, N, triton_ms, torch_ms, speedup
    done_when: python kernels/softmax/bench_softmax.py 输出完整的性能对比表格
    time_minutes: 30
    depends_on: []
    refs:
      - title: Triton benchmark 工具
        url: https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html

  compare_torch:
    summary: 对比 Triton 实现与 PyTorch 原生 softmax 的性能
    steps:
      - 在 bench_softmax.py 中添加 torch.softmax 的 benchmark
      - 计算 speedup = torch_ms / triton_ms
      - 找出 Triton 比 PyTorch 快/慢的尺寸分界点
      - 记录观察到的性能特征到 kernels/softmax/README.md
    done_when: README.md 中有性能对比结论（哪些尺寸 Triton 更快，为什么）
    time_minutes: 20
    depends_on: [write_benchmark]

  compute_gbs:
    summary: 计算 softmax 的有效带宽（GB/s）
    steps:
      - softmax 的理论数据量 = 2 * M * N * sizeof(dtype)（读一次写一次）
      - 计算 achieved_bandwidth = data_bytes / kernel_time
      - 对比 GPU 的理论峰值带宽（查 nvidia-smi 或 deviceQuery）
      - 计算带宽利用率 = achieved / peak * 100%
      - 记录到 README.md
    done_when: README.md 中有带宽利用率数据，能说明 kernel 是否 memory-bound
    time_minutes: 20
    depends_on: [write_benchmark]
    refs:
      - title: GPU 带宽计算方法
        url: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

artifacts:
  benchmark:
    summary: benchmark 脚本和性能数据
    done_when: kernels/softmax/bench_softmax.py 存在且能输出结果
    time_minutes: 0
    depends_on: [write_benchmark, compare_torch, compute_gbs]

total_time_minutes: 70
```

- [ ] **Step 6: Create day05.yaml**

Create `study-plan/guides/day05.yaml`:

```yaml
day: 5

tasks:
  isolate_one_kernel:
    summary: 隔离单次 softmax kernel 调用，准备 profiling
    steps:
      - 创建 kernels/softmax/profile_softmax.py
      - 选择一个代表性尺寸（如 M=2048, N=4096）
      - 用 torch.cuda.synchronize() 包裹单次 kernel 调用
      - 确认脚本只 launch 一次 kernel（避免 warmup 干扰 profile）
    done_when: python kernels/softmax/profile_softmax.py 运行成功且只 launch 一次 kernel
    time_minutes: 15
    depends_on: []

  run_ncu:
    summary: 用 Nsight Compute 采集 softmax kernel 的硬件计数器
    steps:
      - 运行 ncu --set full -o softmax_profile python kernels/softmax/profile_softmax.py
      - 确认生成 softmax_profile.ncu-rep 文件
      - 用 ncu --import softmax_profile.ncu-rep --page raw 查看关键指标
      - 记录 sm__throughput, dram__throughput, l2_hit_rate 等
    done_when: softmax_profile.ncu-rep 存在且能读取硬件计数器数据
    time_minutes: 30
    depends_on: [isolate_one_kernel]
    refs:
      - title: NCU Metrics 参考
        url: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
      - title: Roofline 分析方法
        url: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts

  write_bottleneck_note:
    summary: 分析 profile 数据，写出瓶颈分析笔记
    steps:
      - 判断 kernel 是 compute-bound 还是 memory-bound（对比 arithmetic intensity 与 roofline）
      - 记录主要瓶颈（带宽利用率、occupancy、指令吞吐）
      - 提出可能的优化方向（如 vectorized load、shared memory、kernel fusion）
      - 写入 kernels/softmax/profile-notes.md
    done_when: kernels/softmax/profile-notes.md 存在且包含瓶颈分析和优化建议
    time_minutes: 30
    depends_on: [run_ncu]

artifacts:
  profile:
    summary: ncu profile 文件
    done_when: softmax_profile.ncu-rep 存在
    time_minutes: 0
    depends_on: [run_ncu]
  note:
    summary: 瓶颈分析笔记
    done_when: kernels/softmax/profile-notes.md 存在
    time_minutes: 0
    depends_on: [write_bottleneck_note]

total_time_minutes: 75
```

- [ ] **Step 7: Create day06.yaml**

Create `study-plan/guides/day06.yaml`:

```yaml
day: 6

tasks:
  closed_book_kernel:
    summary: 不看参考代码，从零重写 softmax Triton kernel
    steps:
      - 创建 kernels/softmax/closed_book_softmax.py
      - 凭记忆实现完整的 row_softmax kernel
      - 运行已有的 test_softmax.py 验证正确性
      - 记录卡住的地方和回忆不清的细节
    done_when: closed_book_softmax.py 通过所有已有测试
    time_minutes: 40
    depends_on: []

  explain_masking:
    summary: 用文字解释 Triton softmax 中 mask 的作用和实现
    steps:
      - 在 kernels/softmax/README.md 中添加 "Masking 详解" 章节
      - 解释为什么需要 mask（BLOCK_SIZE > actual N）
      - 解释 mask 在 load 和 store 中的不同用法
      - 解释 mask_value=-inf 对 softmax 数值的影响
      - 画出一个具体例子（N=5, BLOCK_SIZE=8）
    done_when: README.md 中有 Masking 详解章节，包含具体数值例子
    time_minutes: 20
    depends_on: [closed_book_kernel]

  finalize_note:
    summary: 整理 Week 1 softmax 学习笔记，形成完整文档
    steps:
      - 合并 README.md 中的所有章节（寻址、性能、profile、masking）
      - 添加总结段落：softmax kernel 的关键设计决策
      - 添加 "面试要点" 段落：如果被问到 softmax 优化怎么回答
      - 检查文档结构和可读性
    done_when: kernels/softmax/README.md 是一份完整的 softmax kernel 学习笔记
    time_minutes: 25
    depends_on: [explain_masking]

artifacts:
  note:
    summary: 完整的 softmax 学习笔记
    done_when: kernels/softmax/README.md 包含寻址、性能、profile、masking、面试要点
    time_minutes: 0
    depends_on: [finalize_note]

total_time_minutes: 85
```

- [ ] **Step 8: Create day07.yaml**

Create `study-plan/guides/day07.yaml`:

```yaml
day: 7

tasks:
  handwrite_softmax:
    summary: 手写 softmax 公式和 kernel 伪代码（纸笔或白板）
    steps:
      - 写出 softmax 数学公式（含数值稳定版本）
      - 写出 online softmax 的递推公式
      - 画出 Triton kernel 的 grid/block 映射
      - 写出 kernel 伪代码（load, max, sub, exp, sum, div, store）
      - 拍照或扫描保存到 docs/week1-handwrite.jpg
    done_when: docs/week1-handwrite.jpg 存在，内容包含公式和伪代码
    time_minutes: 20
    depends_on: []

  oral_mock:
    summary: 口头模拟面试：讲解 softmax kernel 优化
    steps:
      - 设定场景：面试官问 "请讲一下你做的 softmax kernel 优化"
      - 用 STAR 格式组织回答（Situation/Task/Action/Result）
      - 录音或写下回答要点
      - 自评：是否覆盖了数值稳定性、memory-bound 分析、Triton 实现细节
    done_when: 有一份口头回答的文字记录或录音
    time_minutes: 20
    depends_on: [handwrite_softmax]

  topk_heap_drill:
    summary: 手写 TopK 堆排序算法（coding drill）
    steps:
      - 实现 min-heap 版本的 topk（O(n log k)）
      - 实现 partition 版本的 topk（O(n) 平均）
      - 写出时间复杂度分析
      - 用 3-5 个测试用例验证
    done_when: 两种 topk 实现通过测试，有复杂度分析
    time_minutes: 25
    depends_on: []
    refs:
      - title: TopK 问题总结
        url: https://leetcode.cn/problems/kth-largest-element-in-an-array/

  cpp_raii_move_review:
    summary: 复习 C++ RAII 和 move 语义
    steps:
      - 写一个 RAII 资源管理类（如 FileHandle）
      - 实现 move constructor 和 move assignment
      - 解释 Rule of Five
      - 写出 std::move 和 std::forward 的区别
      - 记录到 docs/cpp-review.md
    done_when: docs/cpp-review.md 存在且包含 RAII 示例和 move 语义解释
    time_minutes: 25
    depends_on: []
    refs:
      - title: C++ Core Guidelines - Resource Management
        url: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r-resource-management

  star_metrics_entry:
    summary: 为 softmax 项目写一条 STAR 格式的项目经历
    steps:
      - Situation：大模型推理中 softmax 是高频算子，PyTorch 默认实现未充分利用 GPU 带宽
      - Task：实现高性能 Triton softmax kernel 并验证性能
      - Action：实现 kernel、编写测试、benchmark、profile 分析
      - Result：填入实际的 speedup 和带宽利用率数据
      - 写入 docs/star-entries.md
    done_when: docs/star-entries.md 存在且有 softmax 项目的 STAR 条目
    time_minutes: 15
    depends_on: [oral_mock]

artifacts:
  mock:
    summary: 模拟面试记录
    done_when: 有口头回答的文字记录
    time_minutes: 0
    depends_on: [oral_mock]

total_time_minutes: 105
```

- [ ] **Step 9: Run full test suite to verify guides load correctly**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add study-plan/guides/
git commit -m "content: add Week 1 daily task guides (day01-day07)"
```

## Task 6: Build, Verify, and Final Commit

**Files:**
- Modify: `study-plan/static/` (rebuilt output)

- [ ] **Step 1: Build frontend**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn/study-plan/frontend
npm run build
```

Expected: build succeeds, output in `study-plan/static/`.

- [ ] **Step 2: Run full frontend test suite**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn/study-plan/frontend
npm test
```

Expected: all tests pass.

- [ ] **Step 3: Run full Python test suite**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn
conda run -n llm-kernel-lab python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 4: Smoke test API with guide data**

Run:

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn
conda run -n llm-kernel-lab python -c "
import sys
sys.path.insert(0, 'study-plan')
import dashboard
dashboard.PROGRESS_FILE = dashboard.BASE_DIR / 'progress.yaml'
data = dashboard.get_api_data()
day1 = data['weeks'][0]['days'][0]
assert 'guide' in day1, 'guide field missing from day 1'
assert day1['guide']['tasks']['audit_existing_kernels']['time_minutes'] == 30
assert day1['guide']['total_time_minutes'] == 105
print('API smoke test PASS')
"
```

Expected: prints "API smoke test PASS".

- [ ] **Step 5: Commit static build**

```bash
cd /home/ycy/code/llm-kernel-lab/.worktrees/dashboard-react-shadcn
git add study-plan/static
git commit -m "build: rebuild React dashboard with GuidedChecklist"
```
