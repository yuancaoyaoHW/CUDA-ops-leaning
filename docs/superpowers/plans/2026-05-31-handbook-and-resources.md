# Implementation Plan: Handbook (sidebar view)

Date: 2026-05-31 (rewritten 2026-05-31 after dashboard-react-shadcn merge decision)

This plan implements the Handbook module described in
`docs/superpowers/specs/2026-05-31-handbook-and-resources-design.md` (revised).

**Resources is no longer in scope.** The original plan also covered an
auto-scanning Resources page; that's been dropped because the
`dashboard-react-shadcn` branch already shipped a manual References CRUD view
that covers the same need (48+ entries already curated). See the spec's
"历史与改名" section.

## Prerequisites

These must be true *before Task 1 starts*:

- [ ] `dashboard-react-shadcn` has been merged into `main`. Verify with:
  `git log main --oneline | grep "merge:.*dashboard-react-shadcn"` returns 1+ lines.
- [ ] `study-plan/frontend/src/components/dashboard/Sidebar.tsx` exists on `main`.
- [ ] `study-plan/static/index.html` exists (built React bundle).
- [ ] `python study-plan/dashboard.py --serve` boots and serves the React UI.
- [ ] `cd study-plan/frontend && npm install && npm run build` exits 0.
- [ ] `pytest tests/test_study_plan_dashboard.py -v` is green.

If any of these fail: do NOT start Task 1. Finish the merge first.

---

## Tech notes / scene

After the merge, the relevant files look like:

- `study-plan/dashboard.py` — Python HTTP server. Serves `/api/dashboard`,
  `/api/save/*`, `/api/references` (CRUD), and the static React bundle from
  `study-plan/static/`. We extend with `/api/handbook` and `/api/handbook/<slug>`.
- `study-plan/frontend/src/components/dashboard/Sidebar.tsx` — left-rail nav,
  7 items (focus / plan / references / operators / libraries / risks / tags).
  We add an 8th: `handbook`.
- `study-plan/frontend/src/components/dashboard/DashboardApp.tsx` — view
  state machine. We add a `view === "handbook"` branch and a
  `viewDescription("handbook")` case.
- `study-plan/frontend/src/api.ts` — fetch helpers (`getDashboard`, `saveDay`,
  `addReference`, etc). We add `listHandbookChapters`, `getHandbookChapter`.
- `study-plan/frontend/src/types.ts` — type definitions. We add
  `HandbookChapterMeta`, `HandbookChapter`.
- `tests/test_study_plan_dashboard.py` — pytest for backend. We add 6 new tests.

---

## Files Created/Modified Summary

| File | Action | Why |
|------|--------|-----|
| `docs/handbook/01-sop.md` | Create | Chapter content (skeleton) |
| `docs/handbook/02-methodology.md` | Create | Chapter content (skeleton) |
| `docs/handbook/03-tools.md` | Create | Chapter content (skeleton) |
| `docs/handbook/04-troubleshooting.md` | Create | Chapter content (skeleton) |
| `study-plan/dashboard.py` | Modify | Add `HANDBOOK_DIR`, `load_handbook`, `/api/handbook*` |
| `tests/test_study_plan_dashboard.py` | Modify | Add 6 handbook tests |
| `study-plan/frontend/package.json` | Modify | Add markdown rendering deps |
| `study-plan/frontend/src/types.ts` | Modify | Add `HandbookChapter*` types |
| `study-plan/frontend/src/api.ts` | Modify | Add 2 fetch helpers |
| `study-plan/frontend/src/components/dashboard/Sidebar.tsx` | Modify | Add `handbook` to View enum + nav |
| `study-plan/frontend/src/components/dashboard/DashboardApp.tsx` | Modify | Render `<HandbookView />` when view==='handbook' |
| `study-plan/frontend/src/components/handbook/HandbookView.tsx` | Create | Top-level: data + layout |
| `study-plan/frontend/src/components/handbook/HandbookNav.tsx` | Create | Chapter list |
| `study-plan/frontend/src/components/handbook/HandbookContent.tsx` | Create | Markdown renderer |
| `study-plan/frontend/src/components/handbook/HandbookContent.test.tsx` | Create | Vitest |

Total: 4 created markdown chapters, 5 created components/tests, 6 modified files.

---

## Task 1: Backend — `load_handbook` + `/api/handbook*`

**Files:**
- Modify: `study-plan/dashboard.py`
- Modify: `tests/test_study_plan_dashboard.py`

TDD: failing tests first, implementation second, then dispatch wiring.

- [ ] **Step 1.1: Add HANDBOOK_DIR constant + HandbookError**

In `study-plan/dashboard.py`, after `BASE_DIR = Path(__file__).parent`:

```python
HANDBOOK_DIR = BASE_DIR.parent / "docs" / "handbook"


class HandbookError(ValueError):
    """Raised when a handbook chapter file is malformed."""
```

- [ ] **Step 1.2: Write failing test — load_handbook returns chapters in order**

In `tests/test_study_plan_dashboard.py`, add to the imports if not already
present:

```python
import importlib.util
from pathlib import Path

def load_dashboard_module():
    spec = importlib.util.spec_from_file_location(
        "dashboard_under_test",
        Path(__file__).resolve().parents[1] / "study-plan" / "dashboard.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

(Skip if a similar helper already exists from earlier work — reuse it.)

Then append:

```python
def test_load_handbook_returns_chapters_in_order(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "02-methodology.md").write_text(
        "---\norder: 2\nslug: methodology\ntitle: 方法论\n---\n# Methodology\n",
        encoding="utf-8",
    )
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\ntitle: SOP\nsubtitle: sub\nicon: play\n---\n# SOP\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    chapters = dashboard.load_handbook()

    assert [c["slug"] for c in chapters] == ["sop", "methodology"]
    assert chapters[0]["title"] == "SOP"
    assert chapters[0]["subtitle"] == "sub"
    assert chapters[0]["icon"] == "play"
    assert chapters[0]["body_md"].startswith("# SOP")
    assert chapters[1]["subtitle"] is None
    assert chapters[1]["icon"] is None
```

Run: `pytest tests/test_study_plan_dashboard.py::test_load_handbook_returns_chapters_in_order -v`
Expected: FAIL with `AttributeError: ... 'load_handbook'`.

- [ ] **Step 1.3: Implement `load_handbook`**

In `study-plan/dashboard.py`:

```python
import re

_FILENAME_RE = re.compile(r"^(\d+)-(.+)\.md$")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def load_handbook() -> list[dict[str, Any]]:
    """Read docs/handbook/NN-slug.md, validate frontmatter, sort by order."""
    if not HANDBOOK_DIR.exists():
        return []
    chapters: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    seen_orders: set[int] = set()
    for path in sorted(HANDBOOK_DIR.iterdir()):
        if path.suffix != ".md":
            continue
        m = _FILENAME_RE.match(path.name)
        if not m:
            raise HandbookError(f"{path.name}: must match NN-slug.md")
        file_slug = m.group(2)
        text = path.read_text(encoding="utf-8")
        fm = _FRONTMATTER_RE.match(text)
        if not fm:
            raise HandbookError(f"{path.name}: missing or unparseable frontmatter")
        meta = yaml.safe_load(fm.group(1)) or {}
        body_md = fm.group(2)
        for required in ("order", "slug", "title"):
            if required not in meta:
                raise HandbookError(f"{path.name}: missing required field '{required}'")
        if meta["slug"] != file_slug:
            raise HandbookError(
                f"{path.name}: frontmatter slug={meta['slug']!r} does not match filename slug={file_slug!r}"
            )
        if meta["slug"] in seen_slugs:
            raise HandbookError(f"duplicate slug: {meta['slug']!r}")
        if meta["order"] in seen_orders:
            raise HandbookError(f"duplicate order: {meta['order']!r}")
        seen_slugs.add(meta["slug"])
        seen_orders.add(meta["order"])
        chapters.append(
            {
                "order": int(meta["order"]),
                "slug": str(meta["slug"]),
                "title": str(meta["title"]),
                "subtitle": meta.get("subtitle"),
                "icon": meta.get("icon"),
                "body_md": body_md,
            }
        )
    chapters.sort(key=lambda c: c["order"])
    return chapters
```

Run the same pytest. Expected: PASS.

- [ ] **Step 1.4: Add validation tests (write all four, then run)**

Append:

```python
def test_load_handbook_rejects_filename_slug_mismatch(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sopx\ntitle: SOP\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    with pytest.raises(dashboard.HandbookError, match="slug"):
        dashboard.load_handbook()


def test_load_handbook_rejects_missing_required_field(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    with pytest.raises(dashboard.HandbookError, match="title"):
        dashboard.load_handbook()


def test_load_handbook_returns_empty_when_dir_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", tmp_path / "nope")
    assert dashboard.load_handbook() == []


def test_load_handbook_keeps_optional_fields_none_when_absent(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\ntitle: SOP\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)
    chapter = dashboard.load_handbook()[0]
    assert chapter["subtitle"] is None
    assert chapter["icon"] is None
```

Run: `pytest tests/test_study_plan_dashboard.py -k load_handbook -v`
Expected: 5 passed.

- [ ] **Step 1.5: Write failing tests for `api_handbook_index` / `api_handbook_chapter`**

Append:

```python
def test_api_handbook_index_omits_body(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\ntitle: SOP\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    payload = dashboard.api_handbook_index()

    assert payload["chapters"][0]["slug"] == "sop"
    assert "body_md" not in payload["chapters"][0]


def test_api_handbook_chapter_returns_body(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\ntitle: SOP\n---\nhello\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    chapter = dashboard.api_handbook_chapter("sop")

    assert chapter is not None
    assert chapter["body_md"].strip() == "hello"


def test_api_handbook_chapter_unknown_slug_returns_none(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    assert dashboard.api_handbook_chapter("does-not-exist") is None
```

Run: FAIL — `AttributeError: ... 'api_handbook_index'`.

- [ ] **Step 1.6: Implement endpoints + dispatch**

Add helpers in `dashboard.py`:

```python
def api_handbook_index() -> dict[str, Any]:
    chapters = load_handbook()
    return {
        "chapters": [
            {k: v for k, v in c.items() if k != "body_md"}
            for c in chapters
        ]
    }


def api_handbook_chapter(slug: str) -> dict[str, Any] | None:
    for chapter in load_handbook():
        if chapter["slug"] == slug:
            return chapter
    return None
```

Then in `DashboardHandler.do_GET` (locate the existing `/api/...` dispatch
chain), add **before** the existing `/api/dashboard` branch:

```python
if parsed.path == "/api/handbook":
    try:
        json_response(self, api_handbook_index())
    except HandbookError as e:
        json_response(self, {"error": str(e)}, status=500)
    return

if parsed.path.startswith("/api/handbook/"):
    slug = parsed.path[len("/api/handbook/"):]
    try:
        chapter = api_handbook_chapter(slug)
    except HandbookError as e:
        json_response(self, {"error": str(e)}, status=500)
        return
    if chapter is None:
        json_response(self, {"error": "not found"}, status=404)
        return
    json_response(self, chapter)
    return
```

(`json_response` already exists on the merged branch — verify by `grep`. If
the helper is named differently, adapt to whatever the merged code uses.)

Run: `pytest tests/test_study_plan_dashboard.py -k handbook -v`
Expected: 8 passed.

- [ ] **Step 1.7: Smoke-test against the running server**

```bash
python study-plan/dashboard.py --serve 8765 &
sleep 1
curl -s http://localhost:8765/api/handbook
curl -s http://localhost:8765/api/handbook/sop
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/api/handbook/nope
kill %1
```

Expected: first two emit JSON (chapters list / single chapter — once Task 2's
files exist; for now `/api/handbook` returns `{"chapters": []}` and the second
returns 404). Third prints `404`.

- [ ] **Step 1.8: Commit**

```bash
git add study-plan/dashboard.py tests/test_study_plan_dashboard.py
git commit -m "feat(dashboard): /api/handbook + /api/handbook/<slug> with frontmatter validation"
```

---

## Task 2: Handbook content skeletons

**Files:**
- Create: `docs/handbook/01-sop.md`
- Create: `docs/handbook/02-methodology.md`
- Create: `docs/handbook/03-tools.md`
- Create: `docs/handbook/04-troubleshooting.md`

These are placeholders with valid frontmatter so the loader (Task 1) and the
frontend (Tasks 4-6) have real chapters to render. The detailed prose is out
of scope for this plan — fill in over time.

- [ ] **Step 2.1: Create `docs/handbook/01-sop.md`**

```markdown
---
order: 1
slug: sop
title: 执行 SOP
subtitle: 一天 / 一周 / 一阶段的闭环动作
icon: play
---

# 执行 SOP

## 一天的闭环

1. `python study-plan/run.py today` 看今天该做什么。
2. 打开当天对应章节的引导（来自 `study-plan/guides/dayNN.yaml`）。
3. 按上午 / 下午 / 晚上的 slot 推进主任务。
4. 当天结束做闭卷日检（15 分钟内手写或口述核心内容）。
5. `python study-plan/run.py day NN done` 跑 strict verify、写回 `progress.yaml`。
6. 在 `notes` 字段补关键卡点 / 明日补课项。

## 一周的闭环

> TODO: 写出周日 mock interview + STAR + drill 的执行步骤。

## 一阶段的闭环

> TODO: 写出第 2 / 4 / 6 / 8 周末完整 mock 的评分流程。
```

- [ ] **Step 2.2: Create `docs/handbook/02-methodology.md`**

```markdown
---
order: 2
slug: methodology
title: 方法论与原则
subtitle: 为什么这么学
icon: brain
---

# 方法论与原则

## JD 拆解到时间分配

> TODO: 把目标岗位的 JD 关键词映射到 kernel / framework / serving / perf / quant / docs / interview 七个维度，并解释 25% / 25% / 15% / 10% / 25% 的分配。

## 三层检验为什么这么设计

> TODO: 解释日检 / 周检 / 阶段检的目的、强度、不通过的处理原则。

## 推荐的学习方式

> TODO: 闭卷手写、托塔（费曼讲解）、benchmark 驱动的复盘。
```

- [ ] **Step 2.3: Create `docs/handbook/03-tools.md`**

(Outer fence is `~~~` to allow inline code fences.)

~~~markdown
---
order: 3
slug: tools
title: 工具手册
subtitle: 怎么用这个仓
icon: wrench
---

# 工具手册

## `study-plan/run.py`

```bash
python study-plan/run.py today              # 看今天
python study-plan/run.py day 5 show         # 看 Day 5
python study-plan/run.py day 5 done         # strict verify + 写回 yaml
python study-plan/run.py week 1 check       # 周检
```

## `study-plan/progress.py` 与 verify 引擎

> TODO: 解释 progress.yaml 为什么不要手改、verify 怎么决定 artifact 真值。

## STAR / drill 解析器

> TODO: STAR.md 与 algorithm-drill.md 的格式约定、被解析后如何写回 progress.yaml。
~~~

- [ ] **Step 2.4: Create `docs/handbook/04-troubleshooting.md`**

```markdown
---
order: 4
slug: troubleshooting
title: 故障排除 + 达标样本
subtitle: 出事怎么办 + 什么叫合格
icon: life-buoy
---

# 故障排除 + 达标样本

## 进度落后

> TODO: 连续 2 天日检不通过 → 当周计划后移 1 天的具体操作。

## GPU / 环境问题

> TODO: ncu 采不到硬件计数器、OOM、占卡不足、WSL2 限制等的排查路径。

## 达标样本

### 一个合格的 kernel

> TODO: vector_add 作为参考——文件清单、tests 覆盖、benchmark 数据点、note 结构。

### 一份合格的 STAR

> TODO: 给一个完整的 STAR 范例（Situation / Task / Action / Result / 反问）。

### 一份合格的 benchmark 报告

> TODO: 给一个 benchmark + nsys + ncu 的产出最小集示例。
```

- [ ] **Step 2.5: Verify the loader accepts all four**

```bash
python -c "
import sys
sys.path.insert(0, 'study-plan')
import dashboard
chapters = dashboard.load_handbook()
print('count =', len(chapters))
for c in chapters:
    print(c['order'], c['slug'], c['title'])
"
```

Expected:
```
count = 4
1 sop 执行 SOP
2 methodology 方法论与原则
3 tools 工具手册
4 troubleshooting 故障排除 + 达标样本
```

- [ ] **Step 2.6: Smoke-test endpoints**

```bash
python study-plan/dashboard.py --serve 8765 &
sleep 1
curl -s http://localhost:8765/api/handbook | python3 -m json.tool | head -20
curl -s http://localhost:8765/api/handbook/sop | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['title']); print(d['body_md'][:120])"
kill %1
```

Expected: lists 4 chapters; second prints "执行 SOP" and the first lines of body.

- [ ] **Step 2.7: Commit**

```bash
git add docs/handbook/
git commit -m "docs(handbook): 4-chapter skeleton (sop/methodology/tools/troubleshooting)"
```

---

## Task 3: Frontend — deps, types, API client

**Files:**
- Modify: `study-plan/frontend/package.json`
- Modify: `study-plan/frontend/src/types.ts`
- Modify: `study-plan/frontend/src/api.ts`
- Modify: `study-plan/frontend/src/main.tsx` (or wherever global CSS imports live)

- [ ] **Step 3.1: Install dependencies**

```bash
cd study-plan/frontend
npm install --save react-markdown@9 remark-gfm@4 \
  rehype-highlight@7 rehype-slug@6 rehype-autolink-headings@7 \
  highlight.js@11
npm install --save-dev @tailwindcss/typography@0.5
```

(Pin to majors so future `npm install` doesn't drift; lockfile pins exact patch versions.)

Verify:

```bash
node -e "
const p = require('./package.json');
const required = {
  'react-markdown': 'dependencies',
  'remark-gfm': 'dependencies',
  'rehype-highlight': 'dependencies',
  'rehype-slug': 'dependencies',
  'rehype-autolink-headings': 'dependencies',
  'highlight.js': 'dependencies',
  '@tailwindcss/typography': 'devDependencies',
};
for (const [k, where] of Object.entries(required)) {
  const v = p[where][k];
  if (!v) throw new Error('missing: ' + k);
  console.log(where, k, '=', v);
}
"
```

- [ ] **Step 3.2: Wire `@tailwindcss/typography` plugin**

Edit `study-plan/frontend/tailwind.config.js` (or `.ts` — whichever the merged
branch uses). In the `plugins` array add `require('@tailwindcss/typography')`.
If the plugins array doesn't exist yet, add `plugins: [require('@tailwindcss/typography')]`.

- [ ] **Step 3.3: Add highlight.js stylesheet import**

In `study-plan/frontend/src/main.tsx`, add **after** the existing tailwind/index.css import:

```ts
import "highlight.js/styles/github.css";
```

- [ ] **Step 3.4: Extend `types.ts`**

Append to `study-plan/frontend/src/types.ts`:

```ts
export interface HandbookChapterMeta {
  order: number;
  slug: string;
  title: string;
  subtitle: string | null;
  icon: string | null;
}

export interface HandbookChapter extends HandbookChapterMeta {
  body_md: string;
}
```

- [ ] **Step 3.5: Extend `api.ts`**

The merged branch's `api.ts` exports named functions (e.g. `getDashboard`,
`saveDay`, `addReference`). Match that style — append:

```ts
import type { HandbookChapter, HandbookChapterMeta } from "./types";

export async function listHandbookChapters(): Promise<HandbookChapterMeta[]> {
  const res = await fetch("/api/handbook");
  if (!res.ok) throw new Error(`GET /api/handbook -> ${res.status}`);
  const data = (await res.json()) as { chapters: HandbookChapterMeta[] };
  return data.chapters;
}

export async function getHandbookChapter(slug: string): Promise<HandbookChapter | null> {
  const res = await fetch(`/api/handbook/${encodeURIComponent(slug)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GET /api/handbook/${slug} -> ${res.status}`);
  return (await res.json()) as HandbookChapter;
}
```

(Verify the existing `api.ts` already uses `fetch` directly, not Axios or
similar — adapt if it does.)

- [ ] **Step 3.6: Type-check**

```bash
cd study-plan/frontend
npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3.7: Commit**

```bash
git add study-plan/frontend/package.json study-plan/frontend/package-lock.json \
  study-plan/frontend/src/types.ts study-plan/frontend/src/api.ts \
  study-plan/frontend/src/main.tsx study-plan/frontend/tailwind.config.*
git commit -m "feat(frontend): handbook deps + types + api helpers"
```

---

## Task 4: Frontend — HandbookContent + HandbookNav

**Files:**
- Create: `study-plan/frontend/src/components/handbook/HandbookContent.tsx`
- Create: `study-plan/frontend/src/components/handbook/HandbookContent.test.tsx`
- Create: `study-plan/frontend/src/components/handbook/HandbookNav.tsx`

- [ ] **Step 4.1: Implement `HandbookContent.tsx`**

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import type { HandbookChapter } from "@/types";

interface Props {
  chapter: HandbookChapter;
}

export function HandbookContent({ chapter }: Props): JSX.Element {
  return (
    <article className="prose prose-slate max-w-none">
      <header className="mb-6 not-prose">
        <h1 className="text-2xl font-semibold text-slate-900">{chapter.title}</h1>
        {chapter.subtitle ? (
          <p className="mt-1 text-sm text-slate-500">{chapter.subtitle}</p>
        ) : null}
      </header>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[
          rehypeSlug,
          [rehypeAutolinkHeadings, { behavior: "wrap" }],
          rehypeHighlight,
        ]}
      >
        {chapter.body_md}
      </ReactMarkdown>
    </article>
  );
}
```

- [ ] **Step 4.2: Vitest spec for `HandbookContent`**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { HandbookContent } from "./HandbookContent";

const chapter = {
  order: 1,
  slug: "sop",
  title: "执行 SOP",
  subtitle: "Sub",
  icon: null,
  body_md: "# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('x')\n```\n",
};

describe("HandbookContent", () => {
  it("renders title and subtitle from frontmatter", () => {
    render(<HandbookContent chapter={chapter} />);
    expect(screen.getByText("执行 SOP")).toBeInTheDocument();
    expect(screen.getByText("Sub")).toBeInTheDocument();
  });

  it("renders GFM tables", () => {
    render(<HandbookContent chapter={chapter} />);
    const table = screen.getByRole("table");
    expect(table).toBeInTheDocument();
    expect(table.querySelectorAll("td").length).toBe(2);
  });

  it("highlights code blocks via rehype-highlight (adds language class)", () => {
    const { container } = render(<HandbookContent chapter={chapter} />);
    const code = container.querySelector("pre code");
    expect(code?.className).toMatch(/language-python|hljs/);
  });

  it("auto-generates heading anchors via rehype-slug", () => {
    const { container } = render(<HandbookContent chapter={chapter} />);
    const heading = container.querySelector("h1");
    expect(heading?.id || heading?.querySelector("[id]")?.id).toBe("heading");
  });
});
```

- [ ] **Step 4.3: Implement `HandbookNav.tsx`**

```tsx
import type { HandbookChapterMeta } from "@/types";

interface Props {
  chapters: HandbookChapterMeta[];
  activeSlug: string;
  onSelect: (slug: string) => void;
}

export function HandbookNav({ chapters, activeSlug, onSelect }: Props): JSX.Element {
  return (
    <nav aria-label="Handbook chapters" className="space-y-1">
      {chapters.map((c) => {
        const active = c.slug === activeSlug;
        return (
          <button
            key={c.slug}
            type="button"
            onClick={() => onSelect(c.slug)}
            aria-current={active ? "page" : undefined}
            className={[
              "block w-full rounded px-3 py-2 text-left text-sm",
              active
                ? "bg-blue-100 text-blue-700"
                : "text-slate-700 hover:bg-slate-100",
            ].join(" ")}
          >
            <span className="font-medium">{c.title}</span>
            {c.subtitle ? <span className="block text-xs opacity-80">{c.subtitle}</span> : null}
          </button>
        );
      })}
    </nav>
  );
}
```

(`bg-blue-100 text-blue-700` matches Sidebar's active state for visual consistency.)

- [ ] **Step 4.4: Build and run vitest**

```bash
cd study-plan/frontend
npx tsc --noEmit
npm run test -- --run HandbookContent
```

Expected: tsc clean; 4 tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add study-plan/frontend/src/components/handbook
git commit -m "feat(frontend): HandbookContent + HandbookNav"
```

---

## Task 5: Frontend — HandbookView + sidebar wiring

**Files:**
- Create: `study-plan/frontend/src/components/handbook/HandbookView.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/Sidebar.tsx`
- Modify: `study-plan/frontend/src/components/dashboard/DashboardApp.tsx`

- [ ] **Step 5.1: Implement `HandbookView.tsx`**

```tsx
import { useEffect, useState } from "react";
import { listHandbookChapters, getHandbookChapter } from "@/api";
import type { HandbookChapter, HandbookChapterMeta } from "@/types";
import { LoadingState } from "@/components/dashboard/LoadingState";
import { EmptyState } from "@/components/dashboard/EmptyState";
import { HandbookNav } from "./HandbookNav";
import { HandbookContent } from "./HandbookContent";

export function HandbookView(): JSX.Element {
  const [chapters, setChapters] = useState<HandbookChapterMeta[] | null>(null);
  const [activeSlug, setActiveSlug] = useState<string>("sop");
  const [chapter, setChapter] = useState<HandbookChapter | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listHandbookChapters()
      .then((cs) => {
        setChapters(cs);
        if (cs.length > 0 && !cs.some((c) => c.slug === activeSlug)) {
          setActiveSlug(cs[0].slug);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Unable to load handbook"));
  }, [activeSlug]);

  useEffect(() => {
    if (!activeSlug) return;
    setChapter(null);
    getHandbookChapter(activeSlug)
      .then(setChapter)
      .catch((e) => setError(e instanceof Error ? e.message : "Unable to load chapter"));
  }, [activeSlug]);

  if (error) {
    return (
      <EmptyState title="Unable To Load Handbook" role="alert">
        <p>{error}</p>
      </EmptyState>
    );
  }
  if (chapters === null) return <LoadingState />;
  if (chapters.length === 0) {
    return (
      <EmptyState title="No Handbook Chapters">
        <p>Add markdown files under <code>docs/handbook/</code> to see them here.</p>
      </EmptyState>
    );
  }

  return (
    <div className="grid gap-6 md:grid-cols-[16rem_1fr]">
      <aside className="md:sticky md:top-0 md:self-start">
        <HandbookNav chapters={chapters} activeSlug={activeSlug} onSelect={setActiveSlug} />
      </aside>
      <section>
        {chapter ? <HandbookContent chapter={chapter} /> : <LoadingState />}
      </section>
    </div>
  );
}
```

(Verify `EmptyState`'s prop API on the merged branch — earlier I saw it accepts
`title` + children. If signature differs, adapt.)

- [ ] **Step 5.2: Add `handbook` to `Sidebar.tsx`**

Edit two lines:

```tsx
// at top
import { CalendarDays, ClipboardList, Cpu, LibraryBig, AlertTriangle, Tags, BookOpen, BookText, RefreshCw } from "lucide-react";

// View enum
export type View = "focus" | "plan" | "operators" | "libraries" | "risks" | "tags" | "references" | "handbook";

// navItems: append at end (after "tags")
const navItems: { id: View; label: string; icon: React.ReactNode }[] = [
  { id: "focus", label: "Focus", icon: <CalendarDays className="h-5 w-5" /> },
  { id: "plan", label: "Plan", icon: <ClipboardList className="h-5 w-5" /> },
  { id: "references", label: "References", icon: <BookOpen className="h-5 w-5" /> },
  { id: "operators", label: "Operators", icon: <Cpu className="h-5 w-5" /> },
  { id: "libraries", label: "Libraries", icon: <LibraryBig className="h-5 w-5" /> },
  { id: "risks", label: "Risks", icon: <AlertTriangle className="h-5 w-5" /> },
  { id: "tags", label: "Tags", icon: <Tags className="h-5 w-5" /> },
  { id: "handbook", label: "Handbook", icon: <BookText className="h-5 w-5" /> },
];
```

- [ ] **Step 5.3: Add `view === "handbook"` branch to `DashboardApp.tsx`**

In `DashboardApp.tsx`, after the existing `{view === "tags" && (…)}` block,
add:

```tsx
{view === "handbook" && <HandbookView />}
```

And import at the top:

```tsx
import { HandbookView } from "@/components/handbook/HandbookView";
```

In the `viewDescription` switch, add:

```tsx
case "handbook":
  return "执行 SOP、方法论、工具手册、故障排除";
```

- [ ] **Step 5.4: Build the React bundle**

```bash
cd study-plan/frontend
npx tsc --noEmit
npm run test -- --run
npm run build
```

Expected: tsc clean; all vitest tests (existing + Task 4's HandbookContent) pass; build emits to `study-plan/static/`.

- [ ] **Step 5.5: Commit**

```bash
git add study-plan/frontend/src \
  study-plan/static
git commit -m "feat(frontend): HandbookView + sidebar item + bundle rebuild"
```

(Include the rebuilt bundle in the same commit since the branch's convention
is to commit `static/` — verify by `git log --oneline -- study-plan/static/ | head` that prior bundle rebuilds were committed.)

---

## Task 6: End-to-end verification

- [ ] **Step 6.1: Backend tests**

```bash
pytest tests/test_study_plan_dashboard.py -v
```

Expected: all PASS, including the 8 new handbook tests.

- [ ] **Step 6.2: Frontend tests + build**

```bash
cd study-plan/frontend
npm run test -- --run
npx tsc --noEmit
npm run build
```

Expected: green / clean / build succeeds.

- [ ] **Step 6.3: Manual checklist**

```bash
python study-plan/dashboard.py --serve 8765
# open http://localhost:8765/ in browser
```

- [ ] Sidebar shows 8 items, last is **Handbook** with book icon
- [ ] Click Handbook → right pane shows HandbookNav (4 chapters) + first chapter content
- [ ] Click chapter 2/3/4 → content swaps; active item highlighted in nav
- [ ] First-load default chapter = "sop"
- [ ] Switching to another sidebar view (e.g. Focus) and back to Handbook preserves last selected chapter (it doesn't — view state isn't persisted across remount; OK)
- [ ] Code blocks have syntax highlighting (CSS classes from highlight.js applied)
- [ ] H1/H2 headings have `id` attributes; clicking them updates URL hash
- [ ] GFM tables render
- [ ] Resize to ≤640px (mobile): sidebar collapses to `w-16` (icons only); HandbookNav stacks above HandbookContent (or grid collapses to single column — verify)
- [ ] References sidebar item is **still working** (regression check: didn't break the merged branch's view)
- [ ] Focus / Plan / Operators / Libraries / Risks / Tags views still load (broader regression check)

- [ ] **Step 6.4: Negative checks**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/api/handbook/no-such-slug
```

Expected: `404`.

In a separate terminal, deliberately corrupt one chapter then verify behavior:

```bash
sed -i 's/^slug: sop$/slug: WRONG/' docs/handbook/01-sop.md
curl -s http://localhost:8765/api/handbook | python3 -m json.tool | head
# expect: {"error": "..."} with HTTP 500
git checkout -- docs/handbook/01-sop.md
curl -s http://localhost:8765/api/handbook | python3 -m json.tool | head
# expect: chapters array of 4
```

This confirms the validation catches mismatched slugs at runtime.

- [ ] **Step 6.5: Final commit (only if cleanup was needed)**

If everything passes, no further commit. Otherwise:

```bash
git add -A
git commit -m "chore: handbook manual-check fixes"
```

---

## Self-Review

Spec coverage check (against the rewritten spec):

| Spec section | Implementing task |
|---|---|
| Handbook file contract (frontmatter, NN-slug.md) | Task 1.3 (loader + validation), Task 2 (skeleton files) |
| `/api/handbook` + `/api/handbook/<slug>` | Task 1.5–1.6 |
| Sidebar 8th view: handbook | Task 5.2 |
| HandbookView two-column layout | Task 5.1 |
| HandbookContent with markdown plugins | Task 4.1–4.2 |
| `react-markdown` + 5 plugins | Task 3.1, Task 4.1 |
| Tailwind typography for prose styling | Task 3.1–3.2 |
| highlight.js github theme | Task 3.1, 3.3 |
| Loading/error states (LoadingState, EmptyState reuse) | Task 5.1 |
| Pytest backend tests (per spec §测试计划) | Task 1.2, 1.4, 1.5 — total 8 tests |
| Vitest frontend tests | Task 4.2 |
| Manual checklist | Task 6.3 |
| Frontmatter error doesn't crash dashboard startup | Step 1.1 (no module-load call to load_handbook), 1.6 (try/except in handler), Step 6.4 (verified) |
| Distinction from References (different sidebar items, both kept) | Task 5.2 (BookOpen vs BookText icons) |

Coverage gaps: none.

Type consistency:
- `HandbookChapterMeta` / `HandbookChapter` types defined once (Task 3.4), used unchanged in Tasks 4 + 5.
- `listHandbookChapters` / `getHandbookChapter` defined once (Task 3.5), consumed in Task 5.1 with matching signatures.
- Backend `load_handbook` returns dicts with same keys as `HandbookChapter` interface (verified via test in Task 1.2).

What changed from the prior plan version:
- Removed Tasks 3 (Resources backend), 7 (WeekBadge/EmptyState — branch already has EmptyState), 8 (ResourcesPage). 
- Removed router (`react-router-dom`) and TabBar — sidebar handles navigation.
- Task 4 (frontend deps/types/api) shrunk: no router deps, no Resources types, no Resources API.
- Task 5 (was: router) replaced with HandbookView + sidebar wiring.
- Task 6 (was: HandbookPage with router) merged into Task 5.
- Plan total: 6 tasks instead of 9; ~1500 fewer lines.
