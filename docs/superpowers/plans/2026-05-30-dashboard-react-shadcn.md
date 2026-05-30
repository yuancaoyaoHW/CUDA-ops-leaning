# Dashboard React shadcn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the handwritten study-plan dashboard frontend with a Vite + React + TypeScript + Tailwind + shadcn/ui frontend while preserving the current Python API and `progress.yaml` persistence.

**Architecture:** Python remains the local API and static-file server. The React app lives in `study-plan/frontend/`, builds into `study-plan/static/`, and talks to the existing `/api/progress`, `/api/day`, `/api/operator`, and `/api/library` endpoints. The dashboard UI is split into data, layout, plan, insight, and editor components.

**Tech Stack:** Python 3.11, PyYAML, Vite, React, TypeScript, Tailwind CSS, shadcn/ui Radix primitives, lucide-react, sonner, Vitest, Testing Library.

---

## File Structure

Create:

- `study-plan/frontend/package.json` - frontend scripts and dependencies.
- `study-plan/frontend/index.html` - Vite HTML entry.
- `study-plan/frontend/vite.config.ts` - Vite, React, Tailwind, test config, build output to `../static`.
- `study-plan/frontend/tsconfig.json` - TypeScript compiler settings.
- `study-plan/frontend/components.json` - shadcn/ui configuration.
- `study-plan/frontend/src/main.tsx` - React mount entry.
- `study-plan/frontend/src/App.tsx` - app shell.
- `study-plan/frontend/src/index.css` - Tailwind and theme variables.
- `study-plan/frontend/src/lib/utils.ts` - `cn()` utility used by shadcn components.
- `study-plan/frontend/src/types.ts` - API data contracts.
- `study-plan/frontend/src/api.ts` - fetch wrapper and save functions.
- `study-plan/frontend/src/dashboardModel.ts` - filter/status/progress helpers.
- `study-plan/frontend/src/test/setup.ts` - Vitest DOM matchers.
- `study-plan/frontend/src/components/dashboard/*.tsx` - dashboard-specific components.
- `study-plan/frontend/src/components/ui/*` - shadcn-generated primitives.

Modify:

- `study-plan/dashboard.py` - serve `study-plan/static/index.html` and assets; keep API endpoints stable.
- `tests/test_study_plan_dashboard.py` - update HTML/static serving assertions.

Commit build output under `study-plan/static/` so the dashboard remains openable after clone without requiring Node for every user.

Remove or stop using:

- `study-plan/dashboard.js`
- `study-plan/dashboard.css`
- old generated handwritten content in `study-plan/dashboard.html`

---

## Task 1: Frontend Workspace and Smoke Test

**Files:**
- Create: `study-plan/frontend/package.json`
- Create: `study-plan/frontend/index.html`
- Create: `study-plan/frontend/vite.config.ts`
- Create: `study-plan/frontend/tsconfig.json`
- Create: `study-plan/frontend/src/main.tsx`
- Create: `study-plan/frontend/src/App.tsx`
- Create: `study-plan/frontend/src/index.css`
- Create: `study-plan/frontend/src/test/setup.ts`
- Create: `study-plan/frontend/src/App.test.tsx`

- [ ] **Step 1: Create package manifest**

```json
{
  "name": "llm-kernel-lab-dashboard",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1",
    "build": "tsc --noEmit && vite build",
    "test": "vitest --run",
    "test:watch": "vitest",
    "preview": "vite preview --host 127.0.0.1"
  },
  "dependencies": {
    "@radix-ui/react-accordion": "^1.2.0",
    "@radix-ui/react-checkbox": "^1.1.0",
    "@radix-ui/react-dialog": "^1.1.0",
    "@radix-ui/react-progress": "^1.1.0",
    "@radix-ui/react-select": "^2.1.0",
    "@radix-ui/react-separator": "^1.1.0",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "lucide-react": "^0.468.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "sonner": "^1.7.0",
    "tailwind-merge": "^2.5.0"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.0.0",
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/user-event": "^14.5.0",
    "@types/node": "^22.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^5.0.0",
    "jsdom": "^25.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.6.0",
    "vite": "^7.0.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 2: Install frontend dependencies**

Run:

```bash
cd study-plan/frontend
npm install
```

Expected: `package-lock.json` is created and npm exits with code 0. If peer dependency versions changed, prefer the latest compatible versions npm resolves instead of downgrading React manually.

- [ ] **Step 3: Create Vite and TypeScript config**

`study-plan/frontend/vite.config.ts`:

```ts
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
  },
});
```

`study-plan/frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2022"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src", "vite.config.ts"]
}
```

- [ ] **Step 4: Create the smoke test before the implementation**

`study-plan/frontend/src/App.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { App } from "./App";

test("renders the dashboard loading shell", () => {
  render(<App />);
  expect(screen.getByText("Loading Dashboard")).toBeInTheDocument();
});
```

Run:

```bash
cd study-plan/frontend
npm test
```

Expected: FAIL because `src/App.tsx` does not exist yet.

- [ ] **Step 5: Add minimal Vite app files**

`study-plan/frontend/index.html`:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#f8fafc" />
    <title>LLM Kernel Lab Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`study-plan/frontend/src/test/setup.ts`:

```ts
import "@testing-library/jest-dom/vitest";
```

`study-plan/frontend/src/index.css`:

```css
@import "tailwindcss";

:root {
  color-scheme: light;
  font-family:
    ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
  background: #f8fafc;
  color: #172033;
  font-synthesis: none;
  -webkit-font-smoothing: antialiased;
}

body {
  margin: 0;
  min-width: 320px;
  min-height: 100vh;
}

button,
input,
select,
textarea {
  font: inherit;
}
```

`study-plan/frontend/src/App.tsx`:

```tsx
export function App() {
  return (
    <main className="mx-auto max-w-7xl px-6 py-8">
      <p className="text-sm font-semibold uppercase tracking-wider text-blue-700">
        LLM Kernel Lab
      </p>
      <h1 className="mt-1 text-2xl font-semibold text-slate-950">
        Loading Dashboard
      </h1>
    </main>
  );
}
```

`study-plan/frontend/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./index.css";

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

- [ ] **Step 6: Run tests and build**

Run:

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: smoke test passes; `study-plan/static/index.html` and hashed assets are generated.

- [ ] **Step 7: Commit**

```bash
git add study-plan/frontend study-plan/static
git commit -m "feat: scaffold React dashboard frontend"
```

---

## Task 2: shadcn/ui Base Components

**Files:**
- Create: `study-plan/frontend/components.json`
- Create: `study-plan/frontend/src/lib/utils.ts`
- Create/Generate: `study-plan/frontend/src/components/ui/*`
- Modify: `study-plan/frontend/src/index.css`

- [ ] **Step 1: Add shadcn config and utility**

`study-plan/frontend/components.json`:

```json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "new-york",
  "rsc": false,
  "tsx": true,
  "tailwind": {
    "css": "src/index.css",
    "baseColor": "neutral",
    "cssVariables": true
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui",
    "lib": "@/lib"
  },
  "iconLibrary": "lucide"
}
```

`study-plan/frontend/src/lib/utils.ts`:

```ts
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

- [ ] **Step 2: Generate shadcn primitives**

Run:

```bash
cd study-plan/frontend
npx shadcn@latest add button card badge progress select input accordion sheet checkbox textarea separator
```

Expected: component files appear under `src/components/ui/`. If the CLI asks to overwrite files, answer no unless the file is generated by this task.

- [ ] **Step 3: Add theme variables to CSS**

Append to `study-plan/frontend/src/index.css`:

```css
@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --color-card: var(--card);
  --color-card-foreground: var(--card-foreground);
  --color-primary: var(--primary);
  --color-primary-foreground: var(--primary-foreground);
  --color-muted: var(--muted);
  --color-muted-foreground: var(--muted-foreground);
  --color-border: var(--border);
  --color-input: var(--input);
  --color-ring: var(--ring);
}

:root {
  --background: #f8fafc;
  --foreground: #172033;
  --card: #ffffff;
  --card-foreground: #172033;
  --primary: #2563eb;
  --primary-foreground: #ffffff;
  --muted: #eef2f7;
  --muted-foreground: #64748b;
  --border: #dbe3ee;
  --input: #d0dae7;
  --ring: #2563eb;
}
```

- [ ] **Step 4: Add a primitive smoke test**

`study-plan/frontend/src/components/ui/button.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { Button } from "./button";

test("renders a shadcn button", () => {
  render(<Button>Refresh</Button>);
  expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
});
```

Run:

```bash
cd study-plan/frontend
npm test
```

Expected: `App.test.tsx` and `button.test.tsx` pass.

- [ ] **Step 5: Commit**

```bash
git add study-plan/frontend
git commit -m "feat: add shadcn dashboard primitives"
```

---

## Task 3: API Types, Client, and Dashboard Model Helpers

**Files:**
- Create: `study-plan/frontend/src/types.ts`
- Create: `study-plan/frontend/src/api.ts`
- Create: `study-plan/frontend/src/api.test.ts`
- Create: `study-plan/frontend/src/dashboardModel.ts`
- Create: `study-plan/frontend/src/dashboardModel.test.ts`

- [ ] **Step 1: Write API client tests**

`study-plan/frontend/src/api.test.ts`:

```ts
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
```

Run:

```bash
cd study-plan/frontend
npm test -- src/api.test.ts
```

Expected: FAIL because `api.ts` and `types.ts` do not exist.

- [ ] **Step 2: Implement types**

`study-plan/frontend/src/types.ts`:

```ts
export type DayStatus = "not_started" | "in_progress" | "done" | "blocked" | "skipped";
export type OperatorStatus =
  | "not_started"
  | "correctness_stage"
  | "benchmark_stage"
  | "profile_stage"
  | "complete"
  | "blocked";
export type LibraryStatus = "not_started" | "in_progress" | "complete" | "blocked";

export type Checklist = Record<string, boolean | string>;

export interface DashboardDay {
  num: number;
  week: number;
  title: string;
  date?: string;
  status: DayStatus;
  daily_check?: number;
  weekly_check_score?: number;
  stage_check_score?: number;
  jd_tags?: string[];
  tasks?: Checklist;
  artifacts?: Checklist;
  verification?: string;
  weaknesses?: string;
  next_fix?: string;
  notes?: string;
  task_done: number;
  task_total: number;
  artifact_done: number;
  artifact_total: number;
  completion_pct: number;
}

export interface DashboardWeek {
  num: number;
  days: DashboardDay[];
}

export interface OperatorInfo {
  status: OperatorStatus;
  artifacts?: Checklist;
  notes?: string;
}

export interface GpuLibraryInfo {
  status: LibraryStatus;
  evidence?: string[];
}

export interface Maturity {
  done: number;
  total: number;
  pct: number;
  status: string;
}

export interface DashboardData {
  meta: { title?: string; [key: string]: unknown };
  weeks: DashboardWeek[];
  operators: Record<string, OperatorInfo>;
  gpu_libraries: Record<string, GpuLibraryInfo>;
  tag_coverage: Record<string, { done: number; planned: number }>;
  operator_maturity: Record<string, Maturity>;
  status_counts: Record<string, number>;
  summary: {
    done_days: number;
    total_days: number;
    done_tasks: number;
    total_tasks: number;
    done_artifacts: number;
    total_artifacts: number;
    average_daily_check: number | null;
    latest_stage_score?: number | null;
  };
  current_day: DashboardDay | null;
  risks: string[];
  options: {
    day_statuses: DayStatus[];
    operator_statuses: OperatorStatus[];
    library_statuses: LibraryStatus[];
    tags: string[];
  };
}

export type DayUpdates = Partial<Pick<
  DashboardDay,
  | "status"
  | "date"
  | "daily_check"
  | "weekly_check_score"
  | "stage_check_score"
  | "verification"
  | "weaknesses"
  | "next_fix"
  | "notes"
>> & {
  tasks?: Record<string, boolean>;
  artifacts?: Record<string, boolean>;
  auto_status?: boolean;
};
```

- [ ] **Step 3: Implement API client**

`study-plan/frontend/src/api.ts`:

```ts
import type { DashboardData, DayUpdates, LibraryStatus, OperatorStatus } from "./types";

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function readJson(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

async function postJson(url: string, payload: unknown): Promise<void> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await readJson(response);
  if (!response.ok || body.ok === false) {
    throw new ApiError(body.error || `Request failed with HTTP ${response.status}`, response.status);
  }
}

export async function getDashboard(): Promise<DashboardData> {
  const response = await fetch("/api/progress");
  if (!response.ok) {
    throw new ApiError(`Unable to load dashboard data: HTTP ${response.status}`, response.status);
  }
  return response.json() as Promise<DashboardData>;
}

export function saveDay(day: number, updates: DayUpdates): Promise<void> {
  return postJson("/api/day", {
    day,
    updates: { ...updates, auto_status: updates.auto_status ?? true },
  });
}

export function saveOperator(
  operator: string,
  updates: { status?: OperatorStatus; artifacts?: Record<string, boolean>; notes?: string },
): Promise<void> {
  return postJson("/api/operator", { operator, updates });
}

export function saveLibrary(
  library: string,
  updates: { status?: LibraryStatus; evidence?: string[] },
): Promise<void> {
  return postJson("/api/library", { library, updates });
}
```

- [ ] **Step 4: Write dashboard model tests**

`study-plan/frontend/src/dashboardModel.test.ts`:

```ts
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
```

- [ ] **Step 5: Implement dashboard model helpers**

`study-plan/frontend/src/dashboardModel.ts`:

```ts
import type { DashboardData, DashboardDay, DashboardWeek } from "./types";

export interface DashboardFilters {
  week: "all" | "current" | string;
  status: "all" | string;
  tag: "all" | string;
  query: string;
}

const STATUS_LABELS: Record<string, string> = {
  not_started: "Not Started",
  in_progress: "In Progress",
  done: "Done",
  blocked: "Blocked",
  skipped: "Skipped",
  correctness_stage: "Correctness",
  benchmark_stage: "Benchmark",
  profile_stage: "Profile",
  complete: "Complete",
};

export function truthy(value: unknown): boolean {
  return value === true || value === "true" || value === "complete";
}

export function formatStatus(status?: string): string {
  return STATUS_LABELS[status || ""] || status || "Unknown";
}

export function flattenDays(data: Pick<DashboardData, "weeks">): DashboardDay[] {
  return data.weeks.flatMap((week) => week.days);
}

export function currentWeek(data: DashboardData): number {
  return data.current_day?.week || 8;
}

export function filterWeeks(data: DashboardData, filters: DashboardFilters): DashboardWeek[] {
  const targetCurrentWeek = currentWeek(data);
  const query = filters.query.trim().toLowerCase();
  return data.weeks
    .map((week) => {
      const days = week.days.filter((day) => {
        if (filters.week !== "all") {
          const targetWeek = filters.week === "current" ? targetCurrentWeek : Number(filters.week);
          if (day.week !== targetWeek) return false;
        }
        if (filters.status !== "all" && day.status !== filters.status) return false;
        if (filters.tag !== "all" && !(day.jd_tags || []).includes(filters.tag)) return false;
        if (query) {
          const text = [
            day.title,
            day.status,
            day.verification,
            day.weaknesses,
            day.next_fix,
            day.notes,
            ...(day.jd_tags || []),
          ]
            .join(" ")
            .toLowerCase();
          if (!text.includes(query)) return false;
        }
        return true;
      });
      return { ...week, days };
    })
    .filter((week) => week.days.length > 0);
}
```

- [ ] **Step 6: Run tests**

Run:

```bash
cd study-plan/frontend
npm test
```

Expected: all frontend tests pass.

- [ ] **Step 7: Commit**

```bash
git add study-plan/frontend/src/types.ts study-plan/frontend/src/api.ts study-plan/frontend/src/api.test.ts study-plan/frontend/src/dashboardModel.ts study-plan/frontend/src/dashboardModel.test.ts
git commit -m "feat: add dashboard frontend data model"
```

---

## Task 4: Dashboard App Loading, Error, and Layout Shell

**Files:**
- Modify: `study-plan/frontend/src/App.tsx`
- Create: `study-plan/frontend/src/components/dashboard/DashboardApp.tsx`
- Create: `study-plan/frontend/src/components/dashboard/DashboardLayout.tsx`
- Create: `study-plan/frontend/src/components/dashboard/LoadingState.tsx`
- Create: `study-plan/frontend/src/components/dashboard/EmptyState.tsx`
- Modify: `study-plan/frontend/src/App.test.tsx`

- [ ] **Step 1: Replace smoke test with loading/error/data tests**

`study-plan/frontend/src/App.test.tsx`:

```tsx
import { render, screen, waitFor } from "@testing-library/react";
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
  expect(screen.getByText(/local dashboard server/i)).toBeInTheDocument();
});
```

Run:

```bash
cd study-plan/frontend
npm test -- src/App.test.tsx
```

Expected: FAIL because `App` still renders only the static loading shell.

- [ ] **Step 2: Add reusable states and layout**

`study-plan/frontend/src/components/dashboard/LoadingState.tsx`:

```tsx
export function LoadingState() {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm" aria-live="polite">
      <h2 className="text-lg font-semibold text-slate-950">Loading Dashboard</h2>
      <p className="mt-1 text-sm text-slate-600">Reading progress data from the local workspace.</p>
    </section>
  );
}
```

`study-plan/frontend/src/components/dashboard/EmptyState.tsx`:

```tsx
interface EmptyStateProps {
  title: string;
  children: React.ReactNode;
}

export function EmptyState({ title, children }: EmptyStateProps) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
      <div className="mt-1 text-sm text-slate-600">{children}</div>
    </section>
  );
}
```

`study-plan/frontend/src/components/dashboard/DashboardLayout.tsx`:

```tsx
import { Button } from "@/components/ui/button";

interface DashboardLayoutProps {
  title: string;
  onRefresh: () => void;
  children: React.ReactNode;
}

export function DashboardLayout({ title, onRefresh, children }: DashboardLayoutProps) {
  return (
    <main className="mx-auto min-h-screen max-w-7xl px-5 py-6 sm:px-7">
      <header className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-blue-700">LLM Kernel Lab</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-normal text-slate-950">{title}</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-600">
            本地可编辑进度面板。启动 Python dashboard server 后可保存到 progress.yaml。
          </p>
        </div>
        <Button variant="outline" onClick={onRefresh}>Refresh Data</Button>
      </header>
      {children}
    </main>
  );
}
```

- [ ] **Step 3: Implement DashboardApp**

`study-plan/frontend/src/components/dashboard/DashboardApp.tsx`:

```tsx
import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { getDashboard } from "@/api";
import type { DashboardData } from "@/types";
import { DashboardLayout } from "./DashboardLayout";
import { EmptyState } from "./EmptyState";
import { LoadingState } from "./LoadingState";

export function DashboardApp() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const next = await getDashboard();
      setData(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load dashboard data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (loading) {
    return (
      <main className="mx-auto max-w-7xl px-5 py-6 sm:px-7">
        <LoadingState />
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="mx-auto max-w-7xl px-5 py-6 sm:px-7">
        <EmptyState title="Unable To Load Dashboard">
          <p>{error}</p>
          <p className="mt-2">Start the local dashboard server before editing or refreshing data.</p>
        </EmptyState>
      </main>
    );
  }

  return (
    <DashboardLayout
      title={data.meta.title || "Study Plan"}
      onRefresh={() => {
        void refresh().then(() => toast.success("Data refreshed"));
      }}
    >
      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-950">Dashboard Loaded</h2>
      </section>
      <Toaster richColors position="bottom-right" />
    </DashboardLayout>
  );
}
```

`study-plan/frontend/src/App.tsx`:

```tsx
import { DashboardApp } from "./components/dashboard/DashboardApp";

export function App() {
  return <DashboardApp />;
}
```

- [ ] **Step 4: Run tests and build**

Run:

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: tests pass and static build succeeds.

- [ ] **Step 5: Commit**

```bash
git add study-plan/frontend
git commit -m "feat: load dashboard data in React app"
```

---

## Task 5: Current Focus and Overview Panels

**Files:**
- Create: `study-plan/frontend/src/components/dashboard/StatusBadge.tsx`
- Create: `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.tsx`
- Create: `study-plan/frontend/src/components/dashboard/ProgressOverview.tsx`
- Create: `study-plan/frontend/src/components/dashboard/CurrentFocusPanel.test.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/DashboardApp.tsx`

- [ ] **Step 1: Write component tests**

`study-plan/frontend/src/components/dashboard/CurrentFocusPanel.test.tsx`:

```tsx
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
```

Run:

```bash
cd study-plan/frontend
npm test -- src/components/dashboard/CurrentFocusPanel.test.tsx
```

Expected: FAIL because components do not exist.

- [ ] **Step 2: Implement StatusBadge**

`study-plan/frontend/src/components/dashboard/StatusBadge.tsx`:

```tsx
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { formatStatus } from "@/dashboardModel";

const tone: Record<string, string> = {
  done: "border-emerald-200 bg-emerald-50 text-emerald-700",
  in_progress: "border-amber-200 bg-amber-50 text-amber-700",
  blocked: "border-red-200 bg-red-50 text-red-700",
  skipped: "border-red-200 bg-red-50 text-red-700",
  not_started: "border-slate-200 bg-slate-100 text-slate-600",
};

export function StatusBadge({ status }: { status?: string }) {
  return (
    <Badge variant="outline" className={cn("rounded-full font-medium", tone[status || "not_started"])}>
      {formatStatus(status)}
    </Badge>
  );
}
```

- [ ] **Step 3: Implement CurrentFocusPanel**

`study-plan/frontend/src/components/dashboard/CurrentFocusPanel.tsx`:

```tsx
import { CalendarDays, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { DashboardDay } from "@/types";
import { StatusBadge } from "./StatusBadge";

interface CurrentFocusPanelProps {
  day: DashboardDay | null;
  onEditDay: (day: DashboardDay) => void;
}

export function CurrentFocusPanel({ day, onEditDay }: CurrentFocusPanelProps) {
  if (!day) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <h2 className="text-lg font-semibold">All planned days are complete.</h2>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-blue-700">Next Step</p>
          <h2 className="mt-1 text-lg font-semibold text-slate-950">{day.title}</h2>
          <p className="mt-1 flex items-center gap-2 text-sm text-slate-600">
            <CalendarDays className="h-4 w-4" />
            Day {String(day.num).padStart(2, "0")} · Week {day.week}
          </p>
        </div>
        <Button onClick={() => onEditDay(day)}>Edit Day {String(day.num).padStart(2, "0")}</Button>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex flex-wrap gap-2">
          <StatusBadge status={day.status} />
          {(day.jd_tags || []).map((tag) => (
            <span key={tag} className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
              {tag}
            </span>
          ))}
        </div>
        {day.next_fix ? (
          <p className="text-sm text-slate-700"><span className="font-semibold text-slate-950">Next Fix</span>: {day.next_fix}</p>
        ) : null}
        {day.weaknesses ? (
          <p className="text-sm text-slate-700"><span className="font-semibold text-slate-950">Weakness</span>: {day.weaknesses}</p>
        ) : null}
        <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 sm:grid-cols-4">
          <Metric label="Tasks" value={`${day.task_done}/${day.task_total}`} />
          <Metric label="Artifacts" value={`${day.artifact_done}/${day.artifact_total}`} />
          <Metric label="Daily Check" value={`${day.daily_check || 0}/3`} />
          <Metric label="Progress" value={`${Math.round(day.completion_pct)}%`} />
        </div>
        <Progress value={day.completion_pct} aria-label="Current day progress" />
        <div className="grid gap-4 md:grid-cols-2">
          <Checklist title="Tasks" items={day.tasks || {}} />
          <Checklist title="Artifacts" items={day.artifacts || {}} />
        </div>
      </CardContent>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <strong className="block text-xl font-semibold tabular-nums text-slate-950">{value}</strong>
      <span className="text-xs text-slate-600">{label}</span>
    </div>
  );
}

function Checklist({ title, items }: { title: string; items: Record<string, unknown> }) {
  return (
    <div>
      <p className="mb-2 text-xs font-semibold text-slate-600">{title}</p>
      <ul className="grid gap-1.5">
        {Object.entries(items).map(([key, value]) => (
          <li key={key} className="flex items-start gap-2 text-sm text-slate-700">
            <CheckCircle2 className={value ? "mt-0.5 h-4 w-4 text-emerald-600" : "mt-0.5 h-4 w-4 text-slate-300"} />
            <span>{key}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 4: Implement ProgressOverview**

`study-plan/frontend/src/components/dashboard/ProgressOverview.tsx`:

```tsx
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { DashboardData } from "@/types";

interface ProgressOverviewProps {
  summary: DashboardData["summary"];
}

export function ProgressOverview({ summary }: ProgressOverviewProps) {
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <p className="text-xs font-bold uppercase tracking-wider text-blue-700">Overview</p>
      </CardHeader>
      <CardContent className="grid gap-4">
        <OverviewRow label="Days" done={summary.done_days} total={summary.total_days} />
        <OverviewRow label="Tasks" done={summary.done_tasks} total={summary.total_tasks} />
        <OverviewRow label="Artifacts" done={summary.done_artifacts} total={summary.total_artifacts} />
        <div>
          <span className="text-xs text-slate-600">Avg Daily Check</span>
          <strong className="block text-xl font-semibold tabular-nums text-slate-950">
            {summary.average_daily_check ?? "—"}
          </strong>
        </div>
      </CardContent>
    </Card>
  );
}

function OverviewRow({ label, done, total }: { label: string; done: number; total: number }) {
  const value = total ? Math.round((done / total) * 100) : 0;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-slate-600">{label}</span>
        <strong className="tabular-nums text-slate-950">{done}/{total}</strong>
      </div>
      <Progress value={value} aria-label={`${label} progress`} />
    </div>
  );
}
```

- [ ] **Step 5: Wire panels into DashboardApp**

Replace the loaded-state content in `DashboardApp.tsx` with:

```tsx
<div className="grid gap-4">
  <section className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(280px,0.75fr)]">
    <CurrentFocusPanel day={data.current_day} onEditDay={() => undefined} />
    <ProgressOverview summary={data.summary} />
  </section>
</div>
```

Add imports:

```ts
import { CurrentFocusPanel } from "./CurrentFocusPanel";
import { ProgressOverview } from "./ProgressOverview";
```

- [ ] **Step 6: Run tests and build**

Run:

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: all tests pass and dashboard static build succeeds.

- [ ] **Step 7: Commit**

```bash
git add study-plan/frontend
git commit -m "feat: add dashboard focus and overview panels"
```

---

## Task 6: Filters and Week Plan List

**Files:**
- Create: `study-plan/frontend/src/components/dashboard/PlanFilters.tsx`
- Create: `study-plan/frontend/src/components/dashboard/WeekPlanList.tsx`
- Create: `study-plan/frontend/src/components/dashboard/WeekPlanList.test.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/DashboardApp.tsx`

- [ ] **Step 1: Write plan list tests**

`study-plan/frontend/src/components/dashboard/WeekPlanList.test.tsx`:

```tsx
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
```

Run:

```bash
cd study-plan/frontend
npm test -- src/components/dashboard/WeekPlanList.test.tsx
```

Expected: FAIL because `WeekPlanList` does not exist.

- [ ] **Step 2: Implement PlanFilters**

`study-plan/frontend/src/components/dashboard/PlanFilters.tsx`:

```tsx
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatStatus, type DashboardFilters } from "@/dashboardModel";
import type { DashboardData } from "@/types";

interface PlanFiltersProps {
  data: DashboardData;
  filters: DashboardFilters;
  onChange: (filters: DashboardFilters) => void;
}

export function PlanFilters({ data, filters, onChange }: PlanFiltersProps) {
  return (
    <div className="grid gap-3 md:grid-cols-[130px_150px_130px_minmax(220px,1fr)]">
      <FilterSelect
        label="Week"
        value={filters.week}
        values={["all", "current", ...data.weeks.map((week) => String(week.num))]}
        labels={{ all: "All Weeks", current: "Current Week" }}
        onValueChange={(week) => onChange({ ...filters, week })}
      />
      <FilterSelect
        label="Status"
        value={filters.status}
        values={["all", ...data.options.day_statuses]}
        labels={{ all: "All Statuses" }}
        formatter={formatStatus}
        onValueChange={(status) => onChange({ ...filters, status })}
      />
      <FilterSelect
        label="Tag"
        value={filters.tag}
        values={["all", ...data.options.tags]}
        labels={{ all: "All Tags" }}
        onValueChange={(tag) => onChange({ ...filters, tag })}
      />
      <label className="grid gap-1 text-xs font-semibold text-slate-600">
        Search
        <span className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <Input
            className="pl-9"
            value={filters.query}
            placeholder="row_softmax, Nsight, blocked..."
            onChange={(event) => onChange({ ...filters, query: event.target.value })}
          />
        </span>
      </label>
    </div>
  );
}

function FilterSelect({
  label,
  value,
  values,
  labels = {},
  formatter,
  onValueChange,
}: {
  label: string;
  value: string;
  values: string[];
  labels?: Record<string, string>;
  formatter?: (value: string) => string;
  onValueChange: (value: string) => void;
}) {
  return (
    <label className="grid gap-1 text-xs font-semibold text-slate-600">
      {label}
      <Select value={value} onValueChange={onValueChange}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {values.map((item) => (
            <SelectItem key={item} value={item}>
              {labels[item] || formatter?.(item) || (label === "Week" ? `Week ${item}` : item)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  );
}
```

- [ ] **Step 3: Implement WeekPlanList**

`study-plan/frontend/src/components/dashboard/WeekPlanList.tsx`:

```tsx
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Progress } from "@/components/ui/progress";
import type { DashboardDay, DashboardWeek } from "@/types";
import { StatusBadge } from "./StatusBadge";

interface WeekPlanListProps {
  weeks: DashboardWeek[];
  currentWeek: number;
  onEditDay: (day: DashboardDay) => void;
}

export function WeekPlanList({ weeks, currentWeek, onEditDay }: WeekPlanListProps) {
  if (!weeks.length) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
        No matching days. Adjust filters to show planned work.
      </section>
    );
  }

  return (
    <Accordion
      type="multiple"
      defaultValue={weeks.map((week) => `week-${week.num}`).includes(`week-${currentWeek}`) ? [`week-${currentWeek}`] : [`week-${weeks[0].num}`]}
      className="rounded-lg border border-slate-200 bg-white shadow-sm"
    >
      {weeks.map((week) => {
        const done = week.days.filter((day) => day.status === "done").length;
        const percent = week.days.length ? Math.round((done / week.days.length) * 100) : 0;
        return (
          <AccordionItem value={`week-${week.num}`} key={week.num} className="border-slate-200">
            <AccordionTrigger className="px-4 hover:no-underline">
              <span className="flex w-full items-center justify-between gap-4 pr-3">
                <span className="font-semibold text-slate-950">Week {week.num}</span>
                <span className="hidden min-w-36 items-center gap-2 sm:flex">
                  <Progress value={percent} className="h-2" />
                  <span className="text-xs tabular-nums text-slate-600">{percent}%</span>
                </span>
              </span>
            </AccordionTrigger>
            <AccordionContent>
              <div className="divide-y divide-slate-200">
                {week.days.map((day) => (
                  <div key={day.num} className="grid grid-cols-[44px_minmax(0,1fr)_auto] items-center gap-3 px-4 py-3">
                    <strong className="text-sm tabular-nums text-slate-700">D{String(day.num).padStart(2, "0")}</strong>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold text-slate-950">{day.title}</p>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-600">
                        <StatusBadge status={day.status} />
                        <span>{day.task_done}/{day.task_total} tasks</span>
                        <span>{day.artifact_done}/{day.artifact_total} artifacts</span>
                        {(day.jd_tags || []).map((tag) => (
                          <span key={tag} className="rounded-full bg-blue-50 px-2 py-0.5 font-semibold text-blue-700">{tag}</span>
                        ))}
                      </div>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => onEditDay(day)}>
                      Edit
                    </Button>
                  </div>
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        );
      })}
    </Accordion>
  );
}
```

- [ ] **Step 4: Wire filters and plan into DashboardApp**

In `DashboardApp.tsx`, add state:

```tsx
const [filters, setFilters] = useState<DashboardFilters>({
  week: "all",
  status: "all",
  tag: "all",
  query: "",
});
```

Add imports:

```ts
import { filterWeeks, currentWeek, type DashboardFilters } from "@/dashboardModel";
import { PlanFilters } from "./PlanFilters";
import { WeekPlanList } from "./WeekPlanList";
```

Replace the loaded-state content under the focus section with:

```tsx
<section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
  <div className="mb-4 grid gap-3 lg:grid-cols-[minmax(180px,0.45fr)_minmax(0,1.55fr)] lg:items-end">
    <div>
      <h2 className="text-lg font-semibold text-slate-950">Plan</h2>
      <p className="text-sm text-slate-600">Filtered by week, status, tag, and search.</p>
    </div>
    <PlanFilters data={data} filters={filters} onChange={setFilters} />
  </div>
  <WeekPlanList
    weeks={filterWeeks(data, filters)}
    currentWeek={currentWeek(data)}
    onEditDay={() => undefined}
  />
</section>
```

- [ ] **Step 5: Run tests and build**

Run:

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: all frontend tests pass and build succeeds.

- [ ] **Step 6: Commit**

```bash
git add study-plan/frontend
git commit -m "feat: add dashboard plan filters and list"
```

---

## Task 7: Insight Rail

**Files:**
- Create: `study-plan/frontend/src/components/dashboard/InsightRail.tsx`
- Create: `study-plan/frontend/src/components/dashboard/InsightRail.test.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/DashboardApp.tsx`

- [ ] **Step 1: Write insight rail tests**

`study-plan/frontend/src/components/dashboard/InsightRail.test.tsx`:

```tsx
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
} as DashboardData;

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
```

- [ ] **Step 2: Implement InsightRail**

`study-plan/frontend/src/components/dashboard/InsightRail.tsx`:

```tsx
import { AlertTriangle, Cpu, LibraryBig, Tags } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { DashboardData } from "@/types";
import { StatusBadge } from "./StatusBadge";

interface InsightRailProps {
  data: DashboardData;
  onEditOperator: (name: string) => void;
  onEditLibrary: (name: string) => void;
}

export function InsightRail({ data, onEditOperator, onEditLibrary }: InsightRailProps) {
  return (
    <aside className="grid gap-4">
      <Card className="shadow-sm">
        <CardHeader className="flex-row items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <h2 className="text-base font-semibold">Risks</h2>
        </CardHeader>
        <CardContent>
          {data.risks.length ? (
            <ul className="grid gap-2">
              {data.risks.map((risk) => (
                <li key={risk} className="rounded-md border-l-4 border-amber-500 bg-amber-50 px-3 py-2 text-sm text-amber-950">
                  {risk}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-600">No high-priority risks.</p>
          )}
        </CardContent>
      </Card>

      <InsightCard icon={<Cpu className="h-4 w-4" />} title="Operator Maturity">
        {Object.entries(data.operator_maturity).map(([name, item]) => (
          <div key={name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-t border-slate-200 py-3 first:border-t-0 first:pt-0">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{name}</p>
              <p className="text-xs text-slate-600">{item.done}/{item.total} artifacts</p>
              <Progress value={item.pct} className="mt-2 h-2" />
            </div>
            <Button variant="outline" size="sm" onClick={() => onEditOperator(name)} aria-label={`Edit operator ${name}`}>Edit</Button>
          </div>
        ))}
      </InsightCard>

      <InsightCard icon={<LibraryBig className="h-4 w-4" />} title="GPU Library Coverage">
        {Object.entries(data.gpu_libraries).map(([name, info]) => (
          <div key={name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-t border-slate-200 py-3 first:border-t-0 first:pt-0">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{name}</p>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <StatusBadge status={info.status} />
                <span>{(info.evidence || []).length} evidence</span>
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={() => onEditLibrary(name)} aria-label={`Edit library ${name}`}>Edit</Button>
          </div>
        ))}
      </InsightCard>

      <InsightCard icon={<Tags className="h-4 w-4" />} title="JD Tag Coverage">
        {Object.entries(data.tag_coverage).map(([tag, item]) => {
          const percent = item.planned ? Math.round((item.done / item.planned) * 100) : 0;
          return (
            <div key={tag} className="border-t border-slate-200 py-3 first:border-t-0 first:pt-0">
              <div className="mb-2 flex items-center justify-between gap-3">
                <span className="text-sm font-semibold">{tag}</span>
                <span className="text-xs tabular-nums text-slate-600">{item.done}/{item.planned}</span>
              </div>
              <Progress value={percent} className="h-2" />
            </div>
          );
        })}
      </InsightCard>
    </aside>
  );
}

function InsightCard({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <Card className="shadow-sm">
      <CardHeader className="flex-row items-center gap-2">
        <span className="text-blue-700">{icon}</span>
        <h2 className="text-base font-semibold">{title}</h2>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}
```

- [ ] **Step 3: Wire rail into DashboardApp**

Wrap plan section in a workbench grid:

```tsx
<section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
  <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
    ...
  </section>
  <InsightRail
    data={data}
    onEditOperator={() => undefined}
    onEditLibrary={() => undefined}
  />
</section>
```

Add import:

```ts
import { InsightRail } from "./InsightRail";
```

- [ ] **Step 4: Run tests and build**

Run:

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: all frontend tests pass.

- [ ] **Step 5: Commit**

```bash
git add study-plan/frontend
git commit -m "feat: add dashboard insight rail"
```

---

## Task 8: Shared Edit Drawer and Day Save Flow

**Files:**
- Create: `study-plan/frontend/src/components/dashboard/EditDrawer.tsx`
- Create: `study-plan/frontend/src/components/dashboard/EditDrawer.test.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/DashboardApp.tsx`

- [ ] **Step 1: Write day editor tests**

`study-plan/frontend/src/components/dashboard/EditDrawer.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Implement EditDrawer day editor**

`study-plan/frontend/src/components/dashboard/EditDrawer.tsx`:

```tsx
import { FormEvent, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import { formatStatus } from "@/dashboardModel";
import type { DashboardData, DashboardDay, DayUpdates } from "@/types";

export type EditTarget =
  | { type: "day"; day: DashboardDay }
  | { type: "operator"; name: string }
  | { type: "library"; name: string }
  | null;

interface EditDrawerProps {
  open: boolean;
  target: EditTarget;
  options: DashboardData["options"];
  onOpenChange: (open: boolean) => void;
  onSaveDay: (day: number, updates: DayUpdates) => Promise<void>;
  onSaveOperator: (name: string, updates: any) => Promise<void>;
  onSaveLibrary: (name: string, updates: any) => Promise<void>;
}

export function EditDrawer({
  open,
  target,
  options,
  onOpenChange,
  onSaveDay,
}: EditDrawerProps) {
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setError(null);
    setSaving(false);
  }, [target]);

  async function submitDay(event: FormEvent<HTMLFormElement>, day: DashboardDay) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const updates: DayUpdates = {
      status: String(form.get("status") || day.status) as DayUpdates["status"],
      date: String(form.get("date") || ""),
      daily_check: Number(form.get("daily_check") || 0),
      verification: String(form.get("verification") || ""),
      weaknesses: String(form.get("weaknesses") || ""),
      next_fix: String(form.get("next_fix") || ""),
      notes: String(form.get("notes") || ""),
      tasks: readChecks(form, "tasks", day.tasks || {}),
      artifacts: readChecks(form, "artifacts", day.artifacts || {}),
    };
    try {
      setError(null);
      setSaving(true);
      await onSaveDay(day.num, updates);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
        {!target ? null : target.type === "day" ? (
          <form onSubmit={(event) => void submitDay(event, target.day)} className="grid gap-4">
            <SheetHeader>
              <SheetTitle>Day {String(target.day.num).padStart(2, "0")} · {target.day.title}</SheetTitle>
            </SheetHeader>
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
            <Field label="Status">
              <Select name="status" defaultValue={target.day.status}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {options.day_statuses.map((status) => (
                    <SelectItem key={status} value={status}>{formatStatus(status)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field label="Date"><Input name="date" type="date" defaultValue={target.day.date || ""} /></Field>
            <Field label="Daily Check"><Input name="daily_check" type="number" min={0} max={3} defaultValue={target.day.daily_check || 0} /></Field>
            <Checklist name="tasks" title="Tasks" values={target.day.tasks || {}} />
            <Checklist name="artifacts" title="Artifacts" values={target.day.artifacts || {}} />
            <TextField label="Verification" name="verification" value={target.day.verification || ""} />
            <TextField label="Weaknesses" name="weaknesses" value={target.day.weaknesses || ""} />
            <TextField label="Next Fix" name="next_fix" value={target.day.next_fix || ""} />
            <TextField label="Notes" name="notes" value={target.day.notes || ""} />
            <div className="sticky bottom-0 -mx-6 flex justify-end gap-2 border-t bg-white px-6 py-4">
              <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
              <Button type="submit" disabled={saving}>{saving ? "Saving..." : "Save Day"}</Button>
            </div>
          </form>
        ) : (
          <SheetHeader><SheetTitle>Unsupported editor target</SheetTitle></SheetHeader>
        )}
      </SheetContent>
    </Sheet>
  );
}

function readChecks(form: FormData, name: string, source: Record<string, unknown>): Record<string, boolean> {
  const checked = new Set(form.getAll(name).map(String));
  return Object.fromEntries(Object.keys(source).map((key) => [key, checked.has(key)]));
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="grid gap-1 text-sm font-medium text-slate-700">{label}{children}</label>;
}

function TextField({ label, name, value }: { label: string; name: string; value: string }) {
  return (
    <label className="grid gap-1 text-sm font-medium text-slate-700">
      {label}
      <Textarea name={name} defaultValue={value} aria-label={label} />
    </label>
  );
}

function Checklist({ name, title, values }: { name: string; title: string; values: Record<string, unknown> }) {
  return (
    <fieldset className="grid gap-2">
      <legend className="text-sm font-medium text-slate-700">{title}</legend>
      {Object.entries(values).map(([key, value]) => (
        <label key={key} className="flex items-start gap-2 text-sm text-slate-700">
          <Checkbox name={name} value={key} defaultChecked={Boolean(value)} aria-label={key} />
          <span>{key}</span>
        </label>
      ))}
    </fieldset>
  );
}
```

- [ ] **Step 3: Wire day editor into DashboardApp**

Add state and imports:

```tsx
import { saveDay, saveLibrary, saveOperator } from "@/api";
import { EditDrawer, type EditTarget } from "./EditDrawer";

const [editTarget, setEditTarget] = useState<EditTarget>(null);
```

Replace the temporary `onEditDay` callbacks:

```tsx
onEditDay={(day) => setEditTarget({ type: "day", day })}
```

Add drawer near `Toaster`:

```tsx
<EditDrawer
  open={editTarget !== null}
  target={editTarget}
  options={data.options}
  onOpenChange={(open) => {
    if (!open) setEditTarget(null);
  }}
  onSaveDay={async (day, updates) => {
    await saveDay(day, updates);
    await refresh();
    toast.success("Day saved");
  }}
  onSaveOperator={saveOperator}
  onSaveLibrary={saveLibrary}
/>
```

- [ ] **Step 4: Run tests and build**

Run:

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: tests pass and build succeeds.

- [ ] **Step 5: Commit**

```bash
git add study-plan/frontend
git commit -m "feat: add day edit drawer"
```

---

## Task 9: Operator and Library Edit Flows

**Files:**
- Modify: `study-plan/frontend/src/components/dashboard/EditDrawer.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/EditDrawer.test.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/DashboardApp.tsx`

- [ ] **Step 1: Add operator and library tests**

Append to `EditDrawer.test.tsx`:

```tsx
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
```

Expected: FAIL because `EditTarget` does not yet carry operator/library data.

- [ ] **Step 2: Extend EditTarget type**

In `EditDrawer.tsx`, replace `EditTarget` with:

```ts
export type EditTarget =
  | { type: "day"; day: DashboardDay }
  | { type: "operator"; name: string; operator: OperatorInfo }
  | { type: "library"; name: string; library: GpuLibraryInfo }
  | null;
```

Add imports:

```ts
import type { DashboardData, DashboardDay, DayUpdates, GpuLibraryInfo, OperatorInfo } from "@/types";
```

- [ ] **Step 3: Add submit handlers**

Add inside `EditDrawer`:

```tsx
async function submitOperator(event: FormEvent<HTMLFormElement>, name: string, operator: OperatorInfo) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    setError(null);
    setSaving(true);
    await onSaveOperator(name, {
      status: String(form.get("status") || operator.status),
      artifacts: readChecks(form, "artifacts", operator.artifacts || {}),
      notes: String(form.get("notes") || ""),
    });
    onOpenChange(false);
  } catch (err) {
    setError(err instanceof Error ? err.message : "Save failed");
  } finally {
    setSaving(false);
  }
}

async function submitLibrary(event: FormEvent<HTMLFormElement>, name: string, library: GpuLibraryInfo) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    setError(null);
    setSaving(true);
    await onSaveLibrary(name, {
      status: String(form.get("status") || library.status),
      evidence: String(form.get("evidence") || "")
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean),
    });
    onOpenChange(false);
  } catch (err) {
    setError(err instanceof Error ? err.message : "Save failed");
  } finally {
    setSaving(false);
  }
}
```

- [ ] **Step 4: Render operator and library forms**

Replace the `"Unsupported editor target"` fallback with operator/library forms:

```tsx
) : target.type === "operator" ? (
  <form onSubmit={(event) => void submitOperator(event, target.name, target.operator)} className="grid gap-4">
    <SheetHeader><SheetTitle>Operator · {target.name}</SheetTitle></SheetHeader>
    {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
    <Field label="Status">
      <Select name="status" defaultValue={target.operator.status}>
        <SelectTrigger><SelectValue /></SelectTrigger>
        <SelectContent>
          {options.operator_statuses.map((status) => (
            <SelectItem key={status} value={status}>{formatStatus(status)}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
    <Checklist name="artifacts" title="Artifacts" values={target.operator.artifacts || {}} />
    <TextField label="Notes" name="notes" value={target.operator.notes || ""} />
    <div className="sticky bottom-0 -mx-6 flex justify-end gap-2 border-t bg-white px-6 py-4">
      <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
      <Button type="submit" disabled={saving}>{saving ? "Saving..." : "Save Operator"}</Button>
    </div>
  </form>
) : (
  <form onSubmit={(event) => void submitLibrary(event, target.name, target.library)} className="grid gap-4">
    <SheetHeader><SheetTitle>GPU Library · {target.name}</SheetTitle></SheetHeader>
    {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
    <Field label="Status">
      <Select name="status" defaultValue={target.library.status}>
        <SelectTrigger><SelectValue /></SelectTrigger>
        <SelectContent>
          {options.library_statuses.map((status) => (
            <SelectItem key={status} value={status}>{formatStatus(status)}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
    <TextField label="Evidence" name="evidence" value={(target.library.evidence || []).join("\n")} />
    <div className="sticky bottom-0 -mx-6 flex justify-end gap-2 border-t bg-white px-6 py-4">
      <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
      <Button type="submit" disabled={saving}>{saving ? "Saving..." : "Save Library"}</Button>
    </div>
  </form>
)
```

- [ ] **Step 5: Wire operator and library targets in DashboardApp**

Replace `InsightRail` handlers:

```tsx
onEditOperator={(name) => {
  const operator = data.operators[name];
  if (operator) setEditTarget({ type: "operator", name, operator });
}}
onEditLibrary={(name) => {
  const library = data.gpu_libraries[name];
  if (library) setEditTarget({ type: "library", name, library });
}}
```

Update drawer save handlers:

```tsx
onSaveOperator={async (name, updates) => {
  await saveOperator(name, updates);
  await refresh();
  toast.success("Operator saved");
}}
onSaveLibrary={async (name, updates) => {
  await saveLibrary(name, updates);
  await refresh();
  toast.success("Library saved");
}}
```

- [ ] **Step 6: Run tests and build**

Run:

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: frontend tests pass and build succeeds.

- [ ] **Step 7: Commit**

```bash
git add study-plan/frontend
git commit -m "feat: add operator and library editors"
```

---

## Task 10: Python Static Serving Integration

**Files:**
- Modify: `study-plan/dashboard.py`
- Modify: `tests/test_study_plan_dashboard.py`

- [ ] **Step 1: Update backend tests for React static shell**

Replace `test_render_dashboard_uses_external_assets_and_accessible_dialog` with:

```python
def test_render_dashboard_uses_react_static_index(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        '<!doctype html><div id="root"></div><script type="module" src="/assets/index.js"></script>',
        encoding="utf-8",
    )
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)
    monkeypatch.setattr(dashboard, "STATIC_DIR", static_dir)
    monkeypatch.setattr(dashboard, "STATIC_INDEX", static_dir / "index.html")

    html = dashboard.render_dashboard()

    assert 'id="root"' in html
    assert "/assets/index.js" in html
    assert 'id="initial-data"' not in html
```

Add static fallback test:

```python
def test_render_dashboard_has_clear_fallback_when_static_index_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    progress_file = tmp_path / "progress.yaml"
    static_dir = tmp_path / "static"
    write_sample_progress(progress_file)
    monkeypatch.setattr(dashboard, "PROGRESS_FILE", progress_file)
    monkeypatch.setattr(dashboard, "STATIC_DIR", static_dir)
    monkeypatch.setattr(dashboard, "STATIC_INDEX", static_dir / "index.html")

    html = dashboard.render_dashboard()

    assert "React dashboard has not been built" in html
    assert "npm run build" in html
```

Run:

```bash
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q
```

Expected: FAIL because `STATIC_DIR` and `STATIC_INDEX` do not exist and old shell is still rendered.

- [ ] **Step 2: Modify dashboard.py constants**

Replace old static constants:

```python
OUTPUT_HTML = BASE_DIR / "dashboard.html"
CSS_FILE = BASE_DIR / "dashboard.css"
JS_FILE = BASE_DIR / "dashboard.js"
```

with:

```python
OUTPUT_HTML = BASE_DIR / "dashboard.html"
STATIC_DIR = BASE_DIR / "static"
STATIC_INDEX = STATIC_DIR / "index.html"
```

- [ ] **Step 3: Replace render_dashboard**

```python
def render_dashboard(embed_data: bool = False) -> str:
    if STATIC_INDEX.exists():
        return STATIC_INDEX.read_text(encoding="utf-8")

    title = html.escape(load_progress().get("meta", {}).get("title", "Study Plan"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #f8fafc; color: #172033; }}
main {{ max-width: 760px; margin: 80px auto; padding: 24px; background: white; border: 1px solid #dbe3ee; border-radius: 8px; }}
code {{ background: #eef2f7; border-radius: 4px; padding: 2px 4px; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<h2>React dashboard has not been built</h2>
<p>Run <code>cd study-plan/frontend && npm install && npm run build</code>, then start <code>python study-plan/dashboard.py --serve</code>.</p>
</main>
</body>
</html>
"""
```

- [ ] **Step 4: Serve static assets**

Add import near the top:

```python
import mimetypes
```

Add helper near `json_response`:

```python
def file_response(handler: SimpleHTTPRequestHandler, path: Path) -> None:
    body = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
```

Update `DashboardHandler.do_GET`:

```python
def do_GET(self) -> None:
    parsed = urlparse(self.path)
    if parsed.path == "/api/progress":
        json_response(self, get_api_data())
        return
    if parsed.path in ("/", "/dashboard.html"):
        body = render_dashboard().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
    if parsed.path.startswith("/assets/"):
        asset = STATIC_DIR / parsed.path.lstrip("/")
        if asset.exists() and asset.is_file():
            file_response(self, asset)
            return
    super().do_GET()
```

- [ ] **Step 5: Update build command semantics**

Replace `build()` with:

```python
def build() -> None:
    html_text = render_dashboard()
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML}")
```

This keeps `python study-plan/dashboard.py --build` compatible while making React static output the source of the shell.

- [ ] **Step 6: Run backend tests**

Run:

```bash
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q
```

Expected: all dashboard tests pass.

- [ ] **Step 7: Commit**

```bash
git add study-plan/dashboard.py tests/test_study_plan_dashboard.py
git commit -m "feat: serve React dashboard static assets"
```

---

## Task 11: End-to-End Build and Local API Smoke

**Files:**
- Modify: `study-plan/dashboard.html`
- Review: `study-plan/static/index.html`
- Review: `study-plan/static/assets/*`

- [ ] **Step 1: Build frontend**

Run:

```bash
cd study-plan/frontend
npm run build
```

Expected: Vite writes `study-plan/static/index.html` and `study-plan/static/assets/...`.

- [ ] **Step 2: Generate compatibility dashboard.html**

Run:

```bash
conda run -n llm-kernel-lab python study-plan/dashboard.py --build
```

Expected: `Generated /home/ycy/code/llm-kernel-lab/study-plan/dashboard.html`.

- [ ] **Step 3: Start server for smoke testing**

Run in a foreground terminal:

```bash
conda run -n llm-kernel-lab python study-plan/dashboard.py --serve 8766
```

Expected: terminal prints `Serving http://127.0.0.1:8766/dashboard.html`.

- [ ] **Step 4: Verify HTML, JS/CSS assets, and API**

In another terminal:

```bash
curl -fsS http://127.0.0.1:8766/dashboard.html | head -n 5
curl -fsS http://127.0.0.1:8766/api/progress | python -m json.tool | head -n 20
```

Expected: first command returns an HTML document from the React build; second command returns dashboard JSON with `meta`, `weeks`, and `summary`.

- [ ] **Step 5: Stop the foreground server**

Press `Ctrl-C` in the server terminal.

- [ ] **Step 6: Commit build integration artifacts**

```bash
git add study-plan/static study-plan/dashboard.html
git commit -m "build: update React dashboard static output"
```

---

## Task 12: Cleanup Old Handwritten Frontend

**Files:**
- Delete: `study-plan/dashboard.js`
- Delete: `study-plan/dashboard.css`
- Modify: tests or docs only if they still reference deleted assets.

- [ ] **Step 1: Search for old asset references**

Run:

```bash
rg -n "dashboard\\.js|dashboard\\.css|initial-data|render_dashboard\\(embed_data" study-plan tests docs README.md
```

Expected: only historical plan/spec references or no results. Active code should not reference old assets.

- [ ] **Step 2: Delete old asset files**

Run:

```bash
rm study-plan/dashboard.js study-plan/dashboard.css
```

Expected: files are removed from the working tree. If they are untracked, `git status` shows deleted only after they had been added earlier; otherwise they simply disappear.

- [ ] **Step 3: Run frontend and backend checks**

Run:

```bash
cd study-plan/frontend && npm test && npm run build
cd ../..
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q
```

Expected: tests and build pass without old assets.

- [ ] **Step 4: Commit cleanup**

```bash
git add -A study-plan/dashboard.js study-plan/dashboard.css study-plan/frontend study-plan/static tests/test_study_plan_dashboard.py
git commit -m "chore: remove handwritten dashboard frontend"
```

---

## Task 13: Full Verification

**Files:**
- No planned source changes.

- [ ] **Step 1: Run frontend verification**

```bash
cd study-plan/frontend
npm test
npm run build
```

Expected: all frontend tests pass; build succeeds.

- [ ] **Step 2: Run Python tests**

```bash
cd /home/ycy/code/llm-kernel-lab
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q
conda run -n llm-kernel-lab python -m pytest tests -q
```

Expected: dashboard tests and full Python test suite pass.

- [ ] **Step 3: Run repository quick verification**

```bash
ENV_NAME=llm-kernel-lab bash scripts/04_verify_all.sh --quick
```

Expected: Python, PyTorch CUDA, Triton import, and Triton vector add pass. Quick mode may skip full pytest and CUDA extension as designed.

- [ ] **Step 4: Check browser visual verification availability**

```bash
command -v chromium || command -v chromium-browser || command -v google-chrome || command -v playwright
```

When one of those commands prints a path, open:

```text
http://127.0.0.1:8766/dashboard.html
```

Verify manually:

- Top focus panel is readable.
- Default plan view shows all weeks.
- Right rail does not squeeze text into table columns.
- Day editor opens, saves, and closes.
- Failed save keeps the drawer open.

When none of those commands prints a path, record in the final implementation note that browser screenshot verification was not run because no browser automation binary was available.

- [ ] **Step 5: Final commit if verification caused generated-file changes**

```bash
git status --short
git add study-plan/static study-plan/dashboard.html
git commit -m "build: refresh verified React dashboard"
```

Skip this commit if `git status --short` shows no generated-file changes.

---

## Self-Review Notes

- Spec coverage: the plan covers Vite/React/TypeScript scaffold, shadcn primitives, backend-static split, component decomposition, data flow, error handling, default `All Weeks`, edit flows, tests, and verification.
- Scope check: this is one subsystem, the study-plan dashboard frontend. It does not include authentication, charts, remote sync, or broader repo redesign.
- Type consistency: `DashboardData`, `DashboardDay`, `DayUpdates`, `OperatorInfo`, and `GpuLibraryInfo` are introduced before use by later tasks.
