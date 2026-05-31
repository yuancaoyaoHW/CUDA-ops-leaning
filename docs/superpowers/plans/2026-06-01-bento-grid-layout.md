# Bento Grid Dashboard Layout Redesign

**Goal:** Replace the current two-row grid layout with a Bento Grid layout that provides clear visual hierarchy, better space utilization, and faster time-to-insight.

**Approach:** Pure CSS Grid with Tailwind classes. No new dependencies. Restructure `DashboardApp.tsx` layout and split `ProgressOverview` into individual KPI tiles.

---

## Current Layout (problems)

```
Row 1: [CurrentFocusPanel 1.6fr] [ProgressOverview 0.75fr]  ← unbalanced
Row 2: [Plan + Filters 1fr]     [InsightRail 360px]         ← inconsistent ratio
```

- CurrentFocusPanel too wide, ProgressOverview too narrow/short
- Two different grid ratios feel disconnected
- InsightRail crammed into fixed 360px, cards stack vertically forever

## New Bento Grid Layout

```
12-column grid, gap-4

┌─────────────────────────┬───────┬───────┬───────┬───────┐
│                         │ Days  │ Tasks │ Arti- │ Daily │
│   Current Focus         │ 0/56  │ 3/190 │ facts │ Check │
│   (hero tile)           │       │       │ 1/92  │  —    │
│   col-span-7, row-span-3│       │       │       │       │
│                         ├───────┴───────┴───────┴───────┤
│   Day 01 · Week 1      │  Risks                         │
│   Guide steps, time,    │  ⚠ row_softmax 还没有闭环      │
│   done criteria...      │  ⚠ GPU 加速库证据不足           │
│                         ├───────────────┬───────────────┤
│                         │ Operators     │ Tag Coverage  │
├─────────────────────────┤ softmax ██ 0/6│ kernel 2/12   │
│   Notes / Next Fix      │ layernorm ... │ perf 1/8      │
│   (col-span-7)          │ (col-span-3)  │ (col-span-2)  │
├─────────────────────────┴───────────────┴───────────────┤
│  Plan: Filters + Week Accordion (full-width, col-span-12)│
└─────────────────────────────────────────────────────────┘
```

**Responsive breakpoints:**
- `xl` (≥1280px): Full 12-col bento as above
- `lg` (≥1024px): 8-col, hero spans 5, KPIs 2x2 on right
- `md` (≥768px): Single column, KPIs in a row of 4, then focus, then insights side-by-side
- `sm` (<768px): Full stack

---

## File Changes

| File | Action |
|------|--------|
| `src/components/dashboard/DashboardApp.tsx` | Replace grid layout with bento grid |
| `src/components/dashboard/ProgressOverview.tsx` | Refactor: export individual `KpiTile` component |
| `src/components/dashboard/InsightRail.tsx` | Split into `RisksTile`, `OperatorsTile`, `TagCoverageTile` |
| `src/components/dashboard/DashboardLayout.tsx` | Simplify (remove inner padding, let bento handle spacing) |
| `src/components/dashboard/CurrentFocusPanel.tsx` | Remove Card wrapper (bento tile IS the card) |
| Tests | Update layout-dependent assertions |

---

## Task 1: Split ProgressOverview into KpiTile

**Files:** `src/components/dashboard/ProgressOverview.tsx`

- [ ] Refactor `ProgressOverview` to export a reusable `KpiTile` component
- [ ] Each KpiTile is a self-contained Card with label, value, progress bar
- [ ] Keep `ProgressOverview` as a convenience wrapper (renders 4 KpiTiles in a fragment)
- [ ] Verify existing tests still pass

```typescript
// New export
export function KpiTile({ label, done, total }: { label: string; done: number; total: number }) {
  const value = total ? Math.round((done / total) * 100) : 0;
  return (
    <Card className="shadow-sm">
      <CardContent className="p-4">
        <span className="text-xs text-slate-600">{label}</span>
        <strong className="block text-2xl font-semibold tabular-nums text-slate-950">
          {done}/{total}
        </strong>
        <Progress value={value} className="mt-2 h-2" aria-label={`${label} progress`} />
      </CardContent>
    </Card>
  );
}
```

---

## Task 2: Split InsightRail into individual tiles

**Files:** `src/components/dashboard/InsightRail.tsx` → keep as-is but also export:
- `RisksTile` (standalone Card)
- `OperatorsTile` (standalone Card)  
- `TagCoverageTile` (standalone Card)
- `LibraryTile` (standalone Card)

- [ ] Extract each section into its own exported component
- [ ] Each takes only the props it needs (not full `DashboardData`)
- [ ] Keep `InsightRail` as a backward-compatible wrapper
- [ ] Verify InsightRail tests still pass

---

## Task 3: Bento Grid Layout in DashboardApp

**Files:** `src/components/dashboard/DashboardApp.tsx`, `src/components/dashboard/DashboardLayout.tsx`

- [ ] Replace the two-section grid in DashboardApp with a single 12-col bento grid
- [ ] Place components with explicit col-span and row-span classes:

```tsx
<div className="grid grid-cols-12 gap-4">
  {/* Hero: Current Focus — left, spans 7 cols, 3 rows */}
  <div className="col-span-12 xl:col-span-7 xl:row-span-3">
    <CurrentFocusPanel ... />
  </div>

  {/* KPI tiles — top-right, 4 tiles in a 2x2 or 1x4 */}
  <div className="col-span-12 xl:col-span-5 grid grid-cols-2 gap-4">
    <KpiTile label="Days" done={s.done_days} total={s.total_days} />
    <KpiTile label="Tasks" done={s.done_tasks} total={s.total_tasks} />
    <KpiTile label="Artifacts" done={s.done_artifacts} total={s.total_artifacts} />
    <KpiTile label="Daily Check" done={s.average_daily_check ?? 0} total={3} />
  </div>

  {/* Risks — mid-right */}
  <div className="col-span-12 xl:col-span-5">
    <RisksTile risks={data.risks} />
  </div>

  {/* Operators + Tags — bottom-right, side by side */}
  <div className="col-span-12 md:col-span-6 xl:col-span-3">
    <OperatorsTile ... />
  </div>
  <div className="col-span-12 md:col-span-6 xl:col-span-2">
    <TagCoverageTile ... />
  </div>

  {/* Plan — full width bottom */}
  <div className="col-span-12">
    <PlanSection ... />
  </div>
</div>
```

- [ ] Simplify DashboardLayout (just header + children, no max-w constraint change needed)
- [ ] Verify responsive behavior at each breakpoint

---

## Task 4: Adjust CurrentFocusPanel styling

**Files:** `src/components/dashboard/CurrentFocusPanel.tsx`

- [ ] The component already renders a `<Card>`, keep it (it becomes the hero tile's visual boundary)
- [ ] Ensure it fills height in the bento cell (`h-full` on the Card)
- [ ] No structural changes needed — just add `className="h-full"` to the outer Card

---

## Task 5: Update tests

**Files:** All test files that reference layout structure

- [ ] `CurrentFocusPanel.test.tsx` — no changes needed (tests content, not layout)
- [ ] `InsightRail.test.tsx` — verify still passes with extracted components
- [ ] `App.test.tsx` — verify renders without error
- [ ] Run full suite: `npm test`

---

## Task 6: Build and verify

- [ ] `npm run build` — no errors
- [ ] Visual check at 1280px, 1024px, 768px, 375px widths
- [ ] Commit all changes
