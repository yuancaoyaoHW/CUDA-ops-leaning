# Dashboard React + shadcn/ui Redesign

Date: 2026-05-30

## Context

The study-plan dashboard is currently implemented as a Python-generated HTML shell plus standalone `dashboard.js` and `dashboard.css`. It works, but the frontend has grown into a large unstructured surface:

- `study-plan/dashboard.js` handles rendering, state, form editing, filtering, API calls, and accessibility behavior.
- `study-plan/dashboard.css` owns all layout and component styling manually.
- The repository has no Node frontend build chain yet.
- The Python server already provides the right persistence model through `progress.yaml` and local API endpoints.

The goal is to improve the dashboard's visual quality, maintainability, and component consistency by introducing a modern React frontend while preserving the current local Python data flow.

## Decision

Use **Vite + React + TypeScript + Tailwind CSS + shadcn/ui** for the dashboard frontend.

Reasons:

- shadcn/ui supports Vite and initializes dependencies, CSS variables, and local component files through its CLI.
- It provides accessible, modern UI primitives without locking the project into a rigid visual identity.
- The copy-into-project model fits this repository: components can be adjusted for a technical learning dashboard without fighting a theme package.
- Tailwind keeps layout and state styling close to components, reducing the current split between imperative rendering and a large CSS file.
- Vite is sufficient for this local tool; Next.js would add routing and server concepts that the dashboard does not need.

HeroUI and Ant Design were considered. HeroUI is visually polished out of the box and built on Tailwind and React Aria, but its library-style abstraction is less flexible for a small custom dashboard. Ant Design is strong for enterprise back offices, but its default product language is heavier and would need more theme work to avoid a generic admin look.

## Architecture

Keep the backend responsibility in Python:

- Read and write `study-plan/progress.yaml`.
- Serve `GET /api/progress`.
- Serve POST endpoints for day, operator, and library updates.
- Serve the built frontend assets.

Add a new frontend workspace:

```text
study-plan/frontend/
  package.json
  vite.config.ts
  tsconfig.json
  src/
    main.tsx
    App.tsx
    api.ts
    types.ts
    components/
      dashboard/
      ui/
```

The Vite build should output static assets into `study-plan/static/`. The Python dashboard server should serve `study-plan/static/index.html` for `/` and `/dashboard.html`, plus hashed JS/CSS assets from the same directory.

The current `study-plan/dashboard.py` should become mostly an API and static-file host. It should stop embedding a full handwritten UI, except for a minimal HTML fallback shell if needed.

## Components

The React frontend should be split into small, named components:

- `DashboardApp`: fetches data, owns loading/error state, triggers refresh after saves.
- `ApiClient`: wraps `/api/progress`, `/api/day`, `/api/operator`, and `/api/library`.
- `Layout`: top bar, responsive page frame, main content columns.
- `CurrentFocusPanel`: current day, next fix, weakness, task/artifact progress.
- `ProgressOverview`: days, tasks, artifacts, and average daily check.
- `PlanFilters`: week, status, tag, and search filters.
- `WeekPlanList`: week accordion and day rows.
- `InsightRail`: risks, operator maturity, GPU library coverage, and JD tag coverage.
- `EditDrawer`: shared editor shell for day, operator, and library updates.

Use shadcn/ui for core primitives:

- `Button`
- `Card`
- `Badge`
- `Progress`
- `Select`
- `Input`
- `Accordion`
- `Sheet`
- `Checkbox`
- `Textarea`
- `sonner` for toast notifications

Use `lucide-react` for icons where useful, such as current-day, search, risk, GPU/library, and progress affordances.

## Visual Direction

The dashboard should feel like an engineering workbench, not a marketing page or decorative SaaS dashboard.

Design rules:

- Use a calm light theme with white and near-gray surfaces.
- Use blue as the primary action color.
- Use status colors sparingly for meaningful progress, warning, blocked, and done states.
- Keep cards flat and purposeful. Avoid nested cards.
- Use dense but readable rows for the 56-day plan.
- Make the default view `All Weeks`, with the current day still emphasized at the top.
- Prioritize scanning: current focus first, plan second, insights third.

The first implementation should not add dark mode, animated charts, authentication, remote sync, or multi-user features.

## Data Flow

On load:

1. React requests `GET /api/progress`.
2. `DashboardApp` stores the returned dashboard payload.
3. Derived frontend state controls filters: `week`, `status`, `tag`, and `q`.

On edit:

1. User opens `EditDrawer`.
2. The drawer owns local form draft state.
3. Save sends the smallest required POST payload to the matching endpoint.
4. On success, the drawer closes and the app re-fetches `GET /api/progress`.
5. On failure, the drawer remains open and the user's draft is preserved.

The frontend should not duplicate backend status derivation. After any successful save, the backend remains the source of truth.

## Error Handling

Initial load failure:

- Show a focused empty state explaining that the local dashboard server may not be running.
- Avoid a blank page.

Save failure:

- Keep the editor open.
- Preserve the user's input.
- Show a toast with the API error or a clear generic failure message.

Static file preview:

- Static preview may show embedded or previously built data.
- Save actions should clearly explain that the dashboard must be served locally to write `progress.yaml`.

Schema tolerance:

- Missing optional fields should render as empty text, zero counts, or empty lists.
- Unexpected API shape should show an error boundary or fallback state rather than crashing the full page.

## Testing

Keep existing Python tests for API behavior and dashboard serving.

Add frontend verification:

- `npm run build`
- `npm test`
- component tests using `vitest` and `@testing-library/react`

Frontend test coverage should include:

- Initial data load displays the current day.
- Default filter shows all weeks.
- Week, status, tag, and search filters update visible days.
- Day edit drawer opens with existing values.
- Saving a day sends the expected payload.
- Save failure keeps the drawer open.
- Operator and library edit flows send the expected endpoint payloads.

Full verification after implementation:

```bash
conda run -n llm-kernel-lab python -m pytest tests/test_study_plan_dashboard.py -q
conda run -n llm-kernel-lab python -m pytest tests -q
ENV_NAME=llm-kernel-lab bash scripts/04_verify_all.sh --quick
```

If frontend files are changed, also run the frontend build and tests from `study-plan/frontend/`.

## Non-Goals

- Do not introduce Next.js.
- Do not add authentication.
- Do not move persistence out of `progress.yaml`.
- Do not download model weights or external model assets.
- Do not redesign the entire repository site.
- Do not implement charts until the core dashboard is clean.

## Migration Approach

Replace the current dashboard frontend in one implementation pass:

- Keep the Python API behavior stable.
- Keep `study-plan/dashboard.py --serve` as the user-facing command.
- Stop relying on handwritten `study-plan/dashboard.js` and `study-plan/dashboard.css` after the React build is wired in.
- Preserve tests for the API and add frontend tests for the new React behavior.
