# Handbook & Resources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new top-level pages to the React dashboard — a 4-chapter Handbook (markdown-driven) and a Resources catalog (auto-derived from `resources/` + `resources/README.md`) — exposed via 3 routes: `/`, `/handbook`, `/resources`.

**Architecture:** Markdown files live in `docs/handbook/`. Python `dashboard.py` adds 2 read-only API groups (`/api/handbook*`, `/api/resources`) that parse those files and the `resources/` directory once at startup. React frontend (already migrated per spec `2026-05-30-dashboard-react-shadcn-design.md`) gains React Router v6, a TabBar, two pages and shared `WeekBadge` / `EmptyState` components.

**Tech Stack:** Python (PyYAML for frontmatter), pytest. React + TypeScript + Tailwind + shadcn/ui (already in place), `react-router-dom`, `react-markdown`, `remark-gfm`, `rehype-highlight`, `rehype-slug`, `rehype-autolink-headings`, Vitest + Testing Library.

**Spec:** `docs/superpowers/specs/2026-05-31-handbook-and-resources-design.md`

---

## Prerequisites

This plan **must not start** until the React migration plan
`docs/superpowers/plans/2026-05-30-dashboard-react-shadcn.md` has been
executed and merged. Verify before Task 1:

- [ ] `study-plan/frontend/src/App.tsx` exists (React app root)
- [ ] `study-plan/frontend/package.json` lists `vite`, `react`, `tailwindcss`, `@radix-ui/*` (shadcn deps)
- [ ] `study-plan/frontend/src/api.ts` (or `ApiClient`) implements `getProgress()`
- [ ] `study-plan/dashboard.py` serves the built React bundle from `study-plan/static/`
- [ ] `npm run build` (in `study-plan/frontend/`) and `pytest tests/test_study_plan_dashboard.py` both green on a clean checkout

If any prerequisite is missing, stop and finish the migration plan first.

---

## File Structure

| Path | Role | New / Modify |
|------|------|--------------|
| `docs/handbook/01-sop.md` | Chapter 1 markdown skeleton | New |
| `docs/handbook/02-methodology.md` | Chapter 2 markdown skeleton | New |
| `docs/handbook/03-tools.md` | Chapter 3 markdown skeleton | New |
| `docs/handbook/04-troubleshooting.md` | Chapter 4 markdown skeleton | New |
| `study-plan/dashboard.py` | Add `HANDBOOK_DIR`, `RESOURCES_DIR`, `load_handbook`, `load_resources`, 3 GET endpoints + static mount | Modify |
| `tests/test_study_plan_dashboard.py` | Tests for handbook + resources loading + endpoints + path traversal | Modify |
| `study-plan/frontend/package.json` | Add 6 new deps | Modify |
| `study-plan/frontend/src/types.ts` | `HandbookChapterMeta`, `HandbookChapter`, `Resource`, `Facets` | Modify |
| `study-plan/frontend/src/api.ts` | Add `listHandbookChapters`, `getHandbookChapter`, `listResources` | Modify |
| `study-plan/frontend/src/routes.tsx` | Router definition | New |
| `study-plan/frontend/src/main.tsx` | Wrap App with `<BrowserRouter>` + use `routes.tsx` | Modify |
| `study-plan/frontend/src/components/layout/Layout.tsx` | Mount `<TabBar>` + `<Outlet />` | Modify |
| `study-plan/frontend/src/components/layout/TabBar.tsx` | 3-tab nav using `<NavLink>` | New |
| `study-plan/frontend/src/components/shared/WeekBadge.tsx` | Reusable W1-W8 badge | New |
| `study-plan/frontend/src/components/shared/EmptyState.tsx` | Reusable empty-list panel | New |
| `study-plan/frontend/src/pages/DashboardPage.tsx` | Existing dashboard content extracted into a page | Modify (existing components moved) |
| `study-plan/frontend/src/pages/HandbookPage.tsx` | Handbook layout shell | New |
| `study-plan/frontend/src/pages/ResourcesPage.tsx` | Resources layout shell | New |
| `study-plan/frontend/src/components/handbook/HandbookNav.tsx` | Sidebar chapter list | New |
| `study-plan/frontend/src/components/handbook/HandbookContent.tsx` | Markdown renderer | New |
| `study-plan/frontend/src/components/resources/ResourceFilters.tsx` | Search + Week + Tag + Type chips | New |
| `study-plan/frontend/src/components/resources/ResourceGroup.tsx` | Group of cards by type | New |
| `study-plan/frontend/src/components/resources/ResourceCard.tsx` | Single resource | New |
| `study-plan/frontend/src/components/handbook/*.test.tsx` | Vitest specs | New |
| `study-plan/frontend/src/components/resources/*.test.tsx` | Vitest specs | New |
| `study-plan/frontend/src/components/layout/TabBar.test.tsx` | Vitest spec | New |

---

## Task 1: Backend — Handbook loader + endpoints

**Files:**
- Modify: `study-plan/dashboard.py`
- Modify: `tests/test_study_plan_dashboard.py`

- [ ] **Step 1.1: Write failing test — load_handbook returns chapters in order**

Add to `tests/test_study_plan_dashboard.py`:

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
        "---\norder: 1\nslug: sop\ntitle: SOP\n---\n# SOP\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    chapters = dashboard.load_handbook()

    assert [c["slug"] for c in chapters] == ["sop", "methodology"]
    assert chapters[0]["title"] == "SOP"
    assert chapters[0]["body_md"].strip() == "# SOP"
```

- [ ] **Step 1.2: Run the test to verify it fails**

Run: `pytest tests/test_study_plan_dashboard.py::test_load_handbook_returns_chapters_in_order -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'HANDBOOK_DIR'`

- [ ] **Step 1.3: Implement `HANDBOOK_DIR` and `load_handbook`**

In `study-plan/dashboard.py`, after the existing `BASE_DIR = Path(__file__).parent` block add:

```python
HANDBOOK_DIR = BASE_DIR.parent / "docs" / "handbook"
HANDBOOK_REQUIRED_FIELDS = ("order", "slug", "title")
HANDBOOK_OPTIONAL_FIELDS = ("subtitle", "icon")
```

Add a function (place above `def get_api_data`):

```python
def load_handbook() -> list[dict[str, Any]]:
    """Parse docs/handbook/*.md frontmatter + body. Sorted by 'order'.
    Raises ValueError on schema violations so the server fails fast at startup."""
    if not HANDBOOK_DIR.exists():
        return []
    chapters: list[dict[str, Any]] = []
    for path in sorted(HANDBOOK_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError(f"{path.name}: missing frontmatter")
        end = text.find("\n---\n", 4)
        if end == -1:
            raise ValueError(f"{path.name}: unterminated frontmatter")
        meta = yaml.safe_load(text[4:end]) or {}
        body = text[end + 5 :]
        for field in HANDBOOK_REQUIRED_FIELDS:
            if field not in meta:
                raise ValueError(f"{path.name}: missing required field '{field}'")
        filename_slug = path.stem.split("-", 1)[-1]
        if meta["slug"] != filename_slug:
            raise ValueError(
                f"{path.name}: slug '{meta['slug']}' must match filename slug '{filename_slug}'"
            )
        chapters.append(
            {
                "order": int(meta["order"]),
                "slug": str(meta["slug"]),
                "title": str(meta["title"]),
                "subtitle": meta.get("subtitle"),
                "icon": meta.get("icon"),
                "body_md": body,
            }
        )
    chapters.sort(key=lambda c: c["order"])
    return chapters
```

- [ ] **Step 1.4: Run the test to verify it passes**

Run: `pytest tests/test_study_plan_dashboard.py::test_load_handbook_returns_chapters_in_order -v`
Expected: PASS

- [ ] **Step 1.5: Add validation tests (write all four, then run)**

Append to `tests/test_study_plan_dashboard.py`:

```python
def test_load_handbook_rejects_filename_slug_mismatch(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-foo.md").write_text(
        "---\norder: 1\nslug: bar\ntitle: T\n---\nbody\n", encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    import pytest
    with pytest.raises(ValueError, match="slug"):
        dashboard.load_handbook()


def test_load_handbook_rejects_missing_required_field(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\nslug: sop\ntitle: T\n---\nbody\n", encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    import pytest
    with pytest.raises(ValueError, match="order"):
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
        "---\norder: 1\nslug: sop\ntitle: T\n---\nbody\n", encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    chapter = dashboard.load_handbook()[0]
    assert chapter["subtitle"] is None
    assert chapter["icon"] is None
```

Run: `pytest tests/test_study_plan_dashboard.py -k load_handbook -v`
Expected: 5 passed.

- [ ] **Step 1.6: Add HTTP endpoint tests (failing)**

Append:

```python
def test_api_handbook_lists_chapters_meta_only(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\ntitle: SOP\nsubtitle: sub\n---\n# SOP\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    payload = dashboard.api_handbook_index()

    assert payload == {
        "chapters": [
            {"order": 1, "slug": "sop", "title": "SOP", "subtitle": "sub", "icon": None},
        ]
    }
    assert "body_md" not in payload["chapters"][0]


def test_api_handbook_chapter_returns_body(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\ntitle: SOP\n---\n# SOP\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    chapter = dashboard.api_handbook_chapter("sop")

    assert chapter is not None
    assert chapter["slug"] == "sop"
    assert chapter["body_md"].startswith("# SOP")


def test_api_handbook_chapter_unknown_slug_returns_none(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    assert dashboard.api_handbook_chapter("does-not-exist") is None
```

Run: `pytest tests/test_study_plan_dashboard.py -k api_handbook -v`
Expected: FAIL — `AttributeError: ... 'api_handbook_index'`.

- [ ] **Step 1.7: Implement helper functions**

In `dashboard.py`, place above `def get_api_data`:

```python
def api_handbook_index() -> dict[str, Any]:
    chapters = load_handbook()
    return {
        "chapters": [
            {
                "order": c["order"],
                "slug": c["slug"],
                "title": c["title"],
                "subtitle": c["subtitle"],
                "icon": c["icon"],
            }
            for c in chapters
        ]
    }


def api_handbook_chapter(slug: str) -> dict[str, Any] | None:
    for chapter in load_handbook():
        if chapter["slug"] == slug:
            return chapter
    return None
```

Run: `pytest tests/test_study_plan_dashboard.py -k api_handbook -v`
Expected: 3 passed.

- [ ] **Step 1.8: Wire `do_GET` to dispatch new routes**

In `DashboardHandler.do_GET` (currently around line 385), add new branches **before** the `("/", "/dashboard.html")` block:

```python
        if parsed.path == "/api/handbook":
            json_response(self, api_handbook_index())
            return
        if parsed.path.startswith("/api/handbook/"):
            slug = parsed.path[len("/api/handbook/") :]
            chapter = api_handbook_chapter(slug)
            if chapter is None:
                json_response(self, {"error": "not found"}, code=404)
            else:
                json_response(self, chapter)
            return
```

- [ ] **Step 1.9: Smoke-test endpoint dispatch via integration test**

Append:

```python
def test_dashboard_handler_routes_handbook(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    handbook_dir = tmp_path / "handbook"
    handbook_dir.mkdir()
    (handbook_dir / "01-sop.md").write_text(
        "---\norder: 1\nslug: sop\ntitle: SOP\n---\n# SOP\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "HANDBOOK_DIR", handbook_dir)

    # Just confirm the helper functions reachable from the handler module produce expected JSON shape.
    payload = dashboard.api_handbook_index()
    assert payload["chapters"][0]["slug"] == "sop"
    chapter = dashboard.api_handbook_chapter("sop")
    assert chapter and "body_md" in chapter
```

Run: `pytest tests/test_study_plan_dashboard.py -v`
Expected: all (existing + new) PASS.

- [ ] **Step 1.10: Commit**

```bash
git add study-plan/dashboard.py tests/test_study_plan_dashboard.py
git commit -m "feat(dashboard): handbook loader + GET /api/handbook[*] endpoints"
```

---

## Task 2: Handbook content skeletons

**Files:**
- Create: `docs/handbook/01-sop.md`
- Create: `docs/handbook/02-methodology.md`
- Create: `docs/handbook/03-tools.md`
- Create: `docs/handbook/04-troubleshooting.md`

These are placeholders with valid frontmatter so the loader (Task 1) and the
frontend (Tasks 6-9) have real chapters to render. The detailed prose is out of
scope for this plan — fill in over time.

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

Run:

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

- [ ] **Step 2.6: Commit**

```bash
git add docs/handbook/
git commit -m "docs(handbook): 4-chapter skeleton (sop/methodology/tools/troubleshooting)"
```

---

## Task 3: Backend — Resources scanner + endpoint + static mount

**Files:**
- Modify: `study-plan/dashboard.py`
- Modify: `tests/test_study_plan_dashboard.py`

This task is split into 3 sub-areas: scanning the filesystem, parsing the
README, and exposing the result over HTTP plus a guarded static mount.

### 3a — Filesystem scan

- [ ] **Step 3.1: Add Resource constants and dataclass**

In `study-plan/dashboard.py`, after `HANDBOOK_DIR` add:

```python
from dataclasses import dataclass, asdict
from typing import Literal

RESOURCES_DIR = BASE_DIR.parent / "resources"
RESOURCE_TYPES: tuple[str, ...] = ("paper", "repo", "tutorial", "manual", "blog")
RESOURCE_TAG_KEYWORDS: dict[str, list[str]] = {
    "kernel": ["FlashAttention", "softmax", "RMSNorm", "kernel", "GEMM", "attention", "Triton", "CUDA"],
    "framework": ["vLLM", "SGLang", "TensorRT", "framework"],
    "serving": ["PagedAttention", "batching", "Orca", "Mooncake", "KV", "serving", "scheduler"],
    "perf": ["Nsight", "profiling", "performance", "Roofline", "Ring Attention"],
    "quant": ["GPTQ", "AWQ", "FP8", "QLoRA", "quantization", "quant"],
    "docs": ["Programming Guide", "Documentation", "Tutorials"],
    "interview": [],
}


@dataclass
class Resource:
    id: str
    type: str
    title: str
    description: str
    weeks: list[int]
    jd_tags: list[str]
    href: str
    available: bool
    year: int | None
```

- [ ] **Step 3.2: Write failing test — empty resources directory**

Add to `tests/test_study_plan_dashboard.py`:

```python
def test_load_resources_empty_when_dir_missing(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    monkeypatch.setattr(dashboard, "RESOURCES_DIR", tmp_path / "nope")

    result = dashboard.load_resources()

    assert result["resources"] == []
    assert result["facets"]["types"] == list(dashboard.RESOURCE_TYPES)
```

Run: `pytest tests/test_study_plan_dashboard.py::test_load_resources_empty_when_dir_missing -v`
Expected: FAIL — `AttributeError: ... 'load_resources'`.

- [ ] **Step 3.3: Write failing test — scans paper PDFs**

Append:

```python
def test_load_resources_scans_paper_pdfs(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    resources_dir = tmp_path / "resources"
    (resources_dir / "papers").mkdir(parents=True)
    (resources_dir / "papers" / "03_flashattention_dao_2022.pdf").write_bytes(b"%PDF")
    (resources_dir / "papers" / "18_attention_is_all_you_need_2017.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(dashboard, "RESOURCES_DIR", resources_dir)

    result = dashboard.load_resources()

    papers = [r for r in result["resources"] if r["type"] == "paper"]
    assert len(papers) == 2
    fa = next(r for r in papers if "flashattention" in r["id"])
    assert fa["year"] == 2022
    assert fa["available"] is True
    assert fa["href"] == "/resources-static/papers/03_flashattention_dao_2022.pdf"
```

Run that test: FAIL.

- [ ] **Step 3.4: Implement filesystem scanner (papers + dirs)**

Add to `dashboard.py`:

```python
import re

_PAPER_FILENAME_RE = re.compile(r"^(?P<num>\d+)_(?P<slug>.+?)_(?P<year>\d{4})\.pdf$")


def _scan_papers() -> list[Resource]:
    papers_dir = RESOURCES_DIR / "papers"
    if not papers_dir.exists():
        return []
    out: list[Resource] = []
    for path in sorted(papers_dir.glob("*.pdf")):
        match = _PAPER_FILENAME_RE.match(path.name)
        if match:
            slug = match.group("slug")
            year = int(match.group("year"))
            stable_id = f"paper:{path.stem}"
            title = slug.replace("_", " ").title()
        else:
            slug = path.stem
            year = None
            stable_id = f"paper:{path.stem}"
            title = path.stem
        out.append(
            Resource(
                id=stable_id,
                type="paper",
                title=title,
                description="",
                weeks=[],
                jd_tags=[],
                href=f"/resources-static/papers/{path.name}",
                available=True,
                year=year,
            )
        )
    return out


def _scan_dir_resources(subdir: str, type_name: str) -> list[Resource]:
    root = RESOURCES_DIR / subdir
    if not root.exists():
        return []
    out: list[Resource] = []
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        title = entry.name
        out.append(
            Resource(
                id=f"{type_name}:{entry.name}",
                type=type_name,
                title=title,
                description="",
                weeks=[],
                jd_tags=[],
                href=f"/resources-static/{subdir}/{entry.name}",
                available=True,
                year=None,
            )
        )
    return out


def load_resources() -> dict[str, Any]:
    """Scan resources/ and parse resources/README.md once. Returns shape:
    {"resources": [...], "facets": {"types": [...], "weeks": [...], "tags": [...]}}.
    """
    items: list[Resource] = []
    items.extend(_scan_papers())
    items.extend(_scan_dir_resources("repos", "repo"))
    items.extend(_scan_dir_resources("tutorials", "tutorial"))
    items.extend(_scan_dir_resources("manuals", "manual"))
    # README parsing fills weeks/jd_tags/description and adds blog-only entries.
    _enrich_with_readme(items)
    weeks_set: set[int] = set()
    tags_set: set[str] = set()
    for r in items:
        weeks_set.update(r.weeks)
        tags_set.update(r.jd_tags)
    return {
        "resources": [asdict(r) for r in items],
        "facets": {
            "types": list(RESOURCE_TYPES),
            "weeks": sorted(weeks_set) if weeks_set else list(range(1, 9)),
            "tags": [t for t in TAG_ORDER if t in tags_set] or TAG_ORDER,
        },
    }


def _enrich_with_readme(items: list[Resource]) -> None:
    """Stub for Step 3.7 — keeps Step 3.5 tests focused on filesystem scan."""
    return
```

Run: `pytest tests/test_study_plan_dashboard.py -k load_resources -v`
Expected: 2 passed.

### 3b — README parsing

- [ ] **Step 3.5: Add README fixture + failing test**

Append:

```python
README_FIXTURE = """\
# 学习资料索引

## 论文清单（按周排列）

### Week 3-4: GEMM / Attention
| # | 论文 | 用途 |
|---|------|------|
| 3 | FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022) | attention kernel 核心 |

### Week 7: 量化
| # | 论文 | 用途 |
|---|------|------|
| 14 | GPTQ: Accurate Post-Training Quantization (Frantar et al., 2023) | GPTQ 量化 |

## GitHub 仓库

| 仓库 | 用途 | 周 |
|------|------|-----|
| openai/triton | Triton 编译器源码 + tutorials | W1-W4 |

## 博客 / 技术文章

| 文章 | 作者 | 用途 |
|------|------|------|
| The FlashAttention CUDA Kernel Line by Line | Stephen Diehl | FlashAttention 实现细节 |
"""


def _setup_resources_with_readme(tmp_path, monkeypatch):
    dashboard = load_dashboard_module()
    resources_dir = tmp_path / "resources"
    (resources_dir / "papers").mkdir(parents=True)
    (resources_dir / "papers" / "03_flashattention_dao_2022.pdf").write_bytes(b"%PDF")
    (resources_dir / "papers" / "14_gptq_frantar_2023.pdf").write_bytes(b"%PDF")
    (resources_dir / "README.md").write_text(README_FIXTURE, encoding="utf-8")
    monkeypatch.setattr(dashboard, "RESOURCES_DIR", resources_dir)
    return dashboard


def test_load_resources_parses_week_ranges(tmp_path, monkeypatch) -> None:
    dashboard = _setup_resources_with_readme(tmp_path, monkeypatch)
    result = dashboard.load_resources()

    fa = next(r for r in result["resources"] if "flashattention" in r["id"])
    gptq = next(r for r in result["resources"] if "gptq" in r["id"])
    assert fa["weeks"] == [3, 4]
    assert gptq["weeks"] == [7]


def test_load_resources_assigns_jd_tags_via_keywords(tmp_path, monkeypatch) -> None:
    dashboard = _setup_resources_with_readme(tmp_path, monkeypatch)
    result = dashboard.load_resources()

    fa = next(r for r in result["resources"] if "flashattention" in r["id"])
    gptq = next(r for r in result["resources"] if "gptq" in r["id"])
    assert "kernel" in fa["jd_tags"]
    assert "quant" in gptq["jd_tags"]


def test_load_resources_marks_unavailable_when_listed_in_readme_only(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    resources_dir = tmp_path / "resources"
    resources_dir.mkdir(parents=True)
    (resources_dir / "README.md").write_text(README_FIXTURE, encoding="utf-8")
    monkeypatch.setattr(dashboard, "RESOURCES_DIR", resources_dir)

    result = dashboard.load_resources()

    not_local = [r for r in result["resources"] if not r["available"]]
    titles = [r["title"] for r in not_local]
    assert any("FlashAttention" in t for t in titles)


def test_load_resources_includes_blog_entries(tmp_path, monkeypatch) -> None:
    dashboard = _setup_resources_with_readme(tmp_path, monkeypatch)
    result = dashboard.load_resources()

    blogs = [r for r in result["resources"] if r["type"] == "blog"]
    assert any("FlashAttention CUDA Kernel" in r["title"] for r in blogs)
    blog = next(r for r in blogs if "FlashAttention CUDA Kernel" in r["title"])
    assert blog["available"] is False  # blogs have no local file
```

Run: `pytest tests/test_study_plan_dashboard.py -k load_resources -v`
Expected: 2 already-passing + 4 new FAILS.

- [ ] **Step 3.6: Implement README parser**

Replace the `_enrich_with_readme` stub in `dashboard.py` with:

```python
_WEEK_RANGE_RE = re.compile(r"###\s+Week\s+(\d+)(?:-(\d+))?")
_PAPER_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(?P<title>.+?)\s*\|\s*(?P<desc>.+?)\s*\|\s*$")
_REPO_ROW_RE = re.compile(r"^\|\s*(?P<repo>[\w./-]+/[\w./-]+)\s*\|\s*(?P<desc>.+?)\s*\|\s*(?P<weeks>[^|]+?)\s*\|\s*$")
_BLOG_ROW_RE = re.compile(r"^\|\s*(?P<title>[^|]+?)\s*\|\s*(?P<author>[^|]+?)\s*\|\s*(?P<desc>[^|]+?)\s*\|\s*$")
_W_RANGE_RE = re.compile(r"W(\d+)(?:-W?(\d+))?")


def _expand_week_range(start: int, end: int | None) -> list[int]:
    return list(range(start, (end if end else start) + 1))


def _infer_tags(text: str) -> list[str]:
    tags: list[str] = []
    for tag, keywords in RESOURCE_TAG_KEYWORDS.items():
        if any(kw.lower() in text.lower() for kw in keywords):
            tags.append(tag)
    return [t for t in TAG_ORDER if t in tags]


def _enrich_with_readme(items: list[Resource]) -> None:
    readme = RESOURCES_DIR / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Section state machine: which top-level section are we in.
    section: str | None = None  # "papers" | "repos" | "blogs" | None
    current_weeks: list[int] = []
    readme_papers: dict[str, dict[str, Any]] = {}  # title-key → meta

    for line in lines:
        if line.startswith("## 论文清单"):
            section = "papers"; continue
        if line.startswith("## GitHub 仓库"):
            section = "repos"; current_weeks = []; continue
        if line.startswith("## 博客 / 技术文章"):
            section = "blogs"; current_weeks = []; continue
        if line.startswith("## "):
            section = None; continue

        if section == "papers":
            wr = _WEEK_RANGE_RE.match(line)
            if wr:
                start = int(wr.group(1)); end = int(wr.group(2)) if wr.group(2) else None
                current_weeks = _expand_week_range(start, end); continue
            m = _PAPER_ROW_RE.match(line)
            if m and "论文" not in m.group("title") and "---" not in m.group("title"):
                readme_papers[m.group("title").strip()] = {
                    "weeks": list(current_weeks),
                    "description": m.group("desc").strip(),
                }
        elif section == "repos":
            m = _REPO_ROW_RE.match(line)
            if m and m.group("repo") != "仓库":
                weeks_field = m.group("weeks")
                wm = _W_RANGE_RE.search(weeks_field)
                weeks = _expand_week_range(int(wm.group(1)), int(wm.group(2)) if wm and wm.group(2) else None) if wm else []
                desc = m.group("desc").strip()
                items.append(
                    Resource(
                        id=f"repo:{m.group('repo')}",
                        type="repo",
                        title=m.group("repo"),
                        description=desc,
                        weeks=weeks,
                        jd_tags=_infer_tags(desc + " " + m.group("repo")),
                        href=f"https://github.com/{m.group('repo')}",
                        available=any(it.id == f"repo:{m.group('repo').split('/')[-1]}" or it.id == f"repo:{m.group('repo').replace('/', '_')}" for it in items),
                        year=None,
                    )
                )
        elif section == "blogs":
            m = _BLOG_ROW_RE.match(line)
            if m and m.group("title").strip() not in {"文章", "---"} and "---" not in m.group("title"):
                title = m.group("title").strip()
                desc = m.group("desc").strip()
                items.append(
                    Resource(
                        id=f"blog:{title[:40]}",
                        type="blog",
                        title=title,
                        description=desc,
                        weeks=[],
                        jd_tags=_infer_tags(title + " " + desc),
                        href="",
                        available=False,
                        year=None,
                    )
                )

    # Backfill papers: match by title substring (FlashAttention, GPTQ, etc.).
    for paper in [r for r in items if r.type == "paper"]:
        for readme_title, meta in readme_papers.items():
            short = readme_title.split(":")[0].strip().lower().replace("-", "")
            paper_slug = paper.id.removeprefix("paper:").lower().replace("_", "")
            if short and short in paper_slug:
                paper.weeks = meta["weeks"]
                paper.description = meta["description"]
                paper.jd_tags = _infer_tags(readme_title + " " + meta["description"])
                break

    # Add README-only papers as unavailable.
    matched_short_keys: set[str] = set()
    for paper in [r for r in items if r.type == "paper"]:
        matched_short_keys.add(paper.id.removeprefix("paper:").split("_", 1)[-1].split("_")[0].lower())
    for readme_title, meta in readme_papers.items():
        short = readme_title.split(":")[0].strip().lower().replace("-", "").replace(" ", "")
        if not any(short.startswith(k) or k.startswith(short[:6]) for k in matched_short_keys):
            items.append(
                Resource(
                    id=f"paper:readme:{readme_title[:40]}",
                    type="paper",
                    title=readme_title,
                    description=meta["description"],
                    weeks=meta["weeks"],
                    jd_tags=_infer_tags(readme_title + " " + meta["description"]),
                    href="",
                    available=False,
                    year=None,
                )
            )
```

Run: `pytest tests/test_study_plan_dashboard.py -k load_resources -v`
Expected: 6 passed.

- [ ] **Step 3.7: Commit 3a + 3b**

```bash
git add study-plan/dashboard.py tests/test_study_plan_dashboard.py
git commit -m "feat(dashboard): resources scanner + README enrichment"
```


### 3c — HTTP endpoint + static mount

- [ ] **Step 3.8: Write failing test for `/api/resources`**

Append:

```python
def test_api_resources_returns_facets(tmp_path, monkeypatch) -> None:
    dashboard = _setup_resources_with_readme(tmp_path, monkeypatch)

    payload = dashboard.api_resources()

    assert "resources" in payload
    assert payload["facets"]["types"] == list(dashboard.RESOURCE_TYPES)
    assert isinstance(payload["facets"]["weeks"], list)
    assert "kernel" in payload["facets"]["tags"] or "quant" in payload["facets"]["tags"]
```

Run: FAIL — `AttributeError: ... 'api_resources'`.

- [ ] **Step 3.9: Implement `api_resources` and dispatch**

In `dashboard.py`, near `api_handbook_*` add:

```python
def api_resources() -> dict[str, Any]:
    return load_resources()
```

In `DashboardHandler.do_GET`, add a branch (before the `("/", "/dashboard.html")` block):

```python
if parsed.path == "/api/resources":
    json_response(self, api_resources())
    return
```

Run: `pytest tests/test_study_plan_dashboard.py::test_api_resources_returns_facets -v`
Expected: PASS.

- [ ] **Step 3.10: Write failing tests for `/resources-static/` mount**

Append:

```python
def test_resources_static_serves_existing_pdf(tmp_path, monkeypatch) -> None:
    """The handler should resolve /resources-static/papers/foo.pdf to RESOURCES_DIR/papers/foo.pdf."""
    dashboard = load_dashboard_module()
    resources_dir = tmp_path / "resources"
    (resources_dir / "papers").mkdir(parents=True)
    pdf = resources_dir / "papers" / "01_test.pdf"
    pdf.write_bytes(b"%PDF-data")
    monkeypatch.setattr(dashboard, "RESOURCES_DIR", resources_dir)

    resolved = dashboard.resolve_resources_static("/resources-static/papers/01_test.pdf")

    assert resolved == pdf


def test_resources_static_blocks_path_traversal(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    resources_dir = tmp_path / "resources"
    resources_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "RESOURCES_DIR", resources_dir)

    assert dashboard.resolve_resources_static("/resources-static/../dashboard.py") is None
    assert dashboard.resolve_resources_static("/resources-static/papers/../../etc/passwd") is None


def test_resources_static_returns_none_for_missing_file(tmp_path, monkeypatch) -> None:
    dashboard = load_dashboard_module()
    resources_dir = tmp_path / "resources"
    resources_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "RESOURCES_DIR", resources_dir)

    assert dashboard.resolve_resources_static("/resources-static/papers/nope.pdf") is None
```

Run: FAIL.

- [ ] **Step 3.11: Implement static-mount resolver**

Add to `dashboard.py`:

```python
import os
import mimetypes

_STATIC_PREFIX = "/resources-static/"


def resolve_resources_static(url_path: str) -> Path | None:
    """Resolve a /resources-static/... URL to a file path inside RESOURCES_DIR.
    Returns None on traversal, missing file, or non-file targets. Path-only
    helper so it can be unit-tested without HTTP plumbing."""
    if not url_path.startswith(_STATIC_PREFIX):
        return None
    rel = url_path[len(_STATIC_PREFIX) :]
    if not rel:
        return None
    candidate = (RESOURCES_DIR / rel).resolve()
    root = RESOURCES_DIR.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate
```

In `DashboardHandler.do_GET`, add (before the static fallback `super().do_GET()`):

```python
if parsed.path.startswith("/resources-static/"):
    target = resolve_resources_static(parsed.path)
    if target is None:
        self.send_error(404)
        return
    ctype, _ = mimetypes.guess_type(target.name)
    body = target.read_bytes()
    self.send_response(200)
    self.send_header("Content-Type", ctype or "application/octet-stream")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)
    return
```

Run: `pytest tests/test_study_plan_dashboard.py -k resources_static -v`
Expected: 3 passed.

- [ ] **Step 3.12: Run the full backend test file**

Run: `pytest tests/test_study_plan_dashboard.py -v`
Expected: all (existing + new) PASS.

- [ ] **Step 3.13: Commit 3c**

```bash
git add study-plan/dashboard.py tests/test_study_plan_dashboard.py
git commit -m "feat(dashboard): /api/resources + guarded /resources-static/ mount"
```

---

## Task 4: Frontend — deps, types, API client

**Files:**
- Modify: `study-plan/frontend/package.json`
- Modify: `study-plan/frontend/src/types.ts`
- Modify: `study-plan/frontend/src/api.ts`

- [ ] **Step 4.1: Install dependencies**

```bash
cd study-plan/frontend
npm install --save react-router-dom@^6 react-markdown@^9 remark-gfm@^4 \
  rehype-highlight@^7 rehype-slug@^6 rehype-autolink-headings@^7
npm install --save-dev @types/react-router-dom
```

After install, verify the listed versions in `package.json` are pinned (no `^` ranges). Run:

```bash
node -e "
const p = require('./package.json');
for (const k of ['react-router-dom','react-markdown','remark-gfm','rehype-highlight','rehype-slug','rehype-autolink-headings']) {
  const v = p.dependencies[k];
  if (!v || v.startsWith('^') || v.startsWith('~')) {
    throw new Error('unpinned: ' + k + '=' + v);
  }
  console.log(k, '=', v);
}
"
```

If any are still unpinned, edit `package.json` to remove the leading `^`/`~` and re-run `npm install`.

- [ ] **Step 4.2: Add highlight.js theme CSS import**

In `study-plan/frontend/src/main.tsx` (or wherever the global CSS imports live), add **after** the existing tailwind import:

```ts
import "highlight.js/styles/github.css";
```

(Note: `highlight.js` is a transitive dependency of `rehype-highlight`. If `npm ls highlight.js` fails to find it, add it explicitly: `npm install --save highlight.js`.)

- [ ] **Step 4.3: Extend `types.ts`**

Append to `study-plan/frontend/src/types.ts`:

```typescript
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

export interface ResourceItem {
  id: string;
  type: "paper" | "repo" | "tutorial" | "manual" | "blog";
  title: string;
  description: string;
  weeks: number[];
  jd_tags: string[];
  href: string;
  available: boolean;
  year: number | null;
}

export interface ResourceFacets {
  types: string[];
  weeks: number[];
  tags: string[];
}

export interface ResourcesPayload {
  resources: ResourceItem[];
  facets: ResourceFacets;
}
```

- [ ] **Step 4.4: Extend `ApiClient`**

In `study-plan/frontend/src/api.ts`, add (next to the existing `getProgress` etc.):

```typescript
import type {
  HandbookChapterMeta,
  HandbookChapter,
  ResourcesPayload,
} from "./types";

// inside class ApiClient { ... }

  async listHandbookChapters(): Promise<HandbookChapterMeta[]> {
    const res = await fetch("/api/handbook");
    if (!res.ok) throw new Error(`GET /api/handbook -> ${res.status}`);
    const data = (await res.json()) as { chapters: HandbookChapterMeta[] };
    return data.chapters;
  }

  async getHandbookChapter(slug: string): Promise<HandbookChapter | null> {
    const res = await fetch(`/api/handbook/${encodeURIComponent(slug)}`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`GET /api/handbook/${slug} -> ${res.status}`);
    return (await res.json()) as HandbookChapter;
  }

  async listResources(): Promise<ResourcesPayload> {
    const res = await fetch("/api/resources");
    if (!res.ok) throw new Error(`GET /api/resources -> ${res.status}`);
    return (await res.json()) as ResourcesPayload;
  }
```

(If the existing `api.ts` exports a singleton object literal instead of a class, mirror that shape — add three async functions and re-export them. Adapt to whatever the React migration plan landed.)

- [ ] **Step 4.5: Type-check**

```bash
cd study-plan/frontend
npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 4.6: Commit**

```bash
git add study-plan/frontend/package.json study-plan/frontend/package-lock.json \
  study-plan/frontend/src/types.ts study-plan/frontend/src/api.ts \
  study-plan/frontend/src/main.tsx
git commit -m "feat(frontend): add router/markdown deps + handbook/resources types & api"
```

---

## Task 5: Frontend — router, Layout integration, TabBar, page shells

**Files:**
- Create: `study-plan/frontend/src/routes.tsx`
- Modify: `study-plan/frontend/src/main.tsx`
- Modify: `study-plan/frontend/src/components/layout/Layout.tsx`
- Create: `study-plan/frontend/src/components/layout/TabBar.tsx`
- Create: `study-plan/frontend/src/components/layout/TabBar.test.tsx`
- Create: `study-plan/frontend/src/pages/DashboardPage.tsx`
- Create: `study-plan/frontend/src/pages/HandbookPage.tsx`
- Create: `study-plan/frontend/src/pages/ResourcesPage.tsx`

- [ ] **Step 5.1: Extract Dashboard content into `DashboardPage.tsx`**

The existing `App.tsx` (from the React migration) renders the dashboard
directly. Move that JSX body into a new file:

```tsx
// study-plan/frontend/src/pages/DashboardPage.tsx
import { CurrentFocusPanel } from "../components/dashboard/CurrentFocusPanel";
import { ProgressOverview } from "../components/dashboard/ProgressOverview";
import { PlanFilters } from "../components/dashboard/PlanFilters";
import { WeekPlanList } from "../components/dashboard/WeekPlanList";
import { InsightRail } from "../components/dashboard/InsightRail";
import { useDashboardData } from "../hooks/useDashboardData";

export function DashboardPage(): JSX.Element {
  const { data, loading, error, refresh } = useDashboardData();
  if (loading) return <p className="text-slate-500">Loading…</p>;
  if (error || !data) return <p className="text-red-600">Failed to load: {error?.message}</p>;
  return (
    <div className="grid gap-6 lg:grid-cols-[3fr_1fr]">
      <div className="space-y-6">
        <CurrentFocusPanel day={data.current_day} />
        <ProgressOverview summary={data.summary} />
        <PlanFilters />
        <WeekPlanList weeks={data.weeks} onRefresh={refresh} />
      </div>
      <InsightRail data={data} />
    </div>
  );
}
```

(Adjust component names and hook name to whatever the migration plan
actually produced — these are placeholders mirroring the migration spec's
component list. The important point: the dashboard's body lives in
`pages/DashboardPage.tsx`, not in `App.tsx`.)

- [ ] **Step 5.2: Create `routes.tsx`**

```tsx
// study-plan/frontend/src/routes.tsx
import { Navigate, createBrowserRouter } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { DashboardPage } from "./pages/DashboardPage";
import { HandbookPage } from "./pages/HandbookPage";
import { ResourcesPage } from "./pages/ResourcesPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "handbook", element: <Navigate to="/handbook/sop" replace /> },
      { path: "handbook/:slug", element: <HandbookPage /> },
      { path: "resources", element: <ResourcesPage /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);
```

- [ ] **Step 5.3: Update `main.tsx` to render the router**

Replace the current `<App />` rendering with:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router-dom";
import { router } from "./routes";
import "./index.css";
import "highlight.js/styles/github.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
```

`App.tsx` may now be unused — delete it once the dashboard renders correctly through `DashboardPage`.

- [ ] **Step 5.4: Update `Layout.tsx` to host the `<TabBar>` and `<Outlet>`**

```tsx
// study-plan/frontend/src/components/layout/Layout.tsx
import { Outlet } from "react-router-dom";
import { TabBar } from "./TabBar";

export function Layout(): JSX.Element {
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">LLM Kernel Lab</p>
            <h1 className="text-lg font-semibold text-slate-900">大模型推理框架/加速 8 周计划</h1>
          </div>
        </div>
        <div className="mx-auto max-w-6xl px-6">
          <TabBar />
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
```

- [ ] **Step 5.5: Create `TabBar.tsx`**

```tsx
// study-plan/frontend/src/components/layout/TabBar.tsx
import { NavLink } from "react-router-dom";

const TABS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/handbook", label: "Handbook", end: false },
  { to: "/resources", label: "Resources", end: false },
];

export function TabBar(): JSX.Element {
  return (
    <nav role="tablist" aria-label="Primary" className="flex gap-1 border-b border-slate-200">
      {TABS.map((t) => (
        <NavLink
          key={t.to}
          to={t.to}
          end={t.end}
          role="tab"
          className={({ isActive }) =>
            [
              "px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors",
              isActive
                ? "border-slate-900 text-slate-900"
                : "border-transparent text-slate-500 hover:text-slate-700",
            ].join(" ")
          }
        >
          {t.label}
        </NavLink>
      ))}
    </nav>
  );
}
```

- [ ] **Step 5.6: Write Vitest spec for TabBar**

```tsx
// study-plan/frontend/src/components/layout/TabBar.test.tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { TabBar } from "./TabBar";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <TabBar />
    </MemoryRouter>,
  );
}

describe("TabBar", () => {
  it("renders three tabs", () => {
    renderAt("/");
    expect(screen.getByRole("tab", { name: "Dashboard" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Handbook" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Resources" })).toBeInTheDocument();
  });

  it("marks Dashboard active at /", () => {
    renderAt("/");
    expect(screen.getByRole("tab", { name: "Dashboard" })).toHaveAttribute("aria-current", "page");
  });

  it("marks Handbook active at /handbook/sop", () => {
    renderAt("/handbook/sop");
    expect(screen.getByRole("tab", { name: "Handbook" })).toHaveAttribute("aria-current", "page");
  });

  it("marks Resources active at /resources?week=3", () => {
    renderAt("/resources?week=3");
    expect(screen.getByRole("tab", { name: "Resources" })).toHaveAttribute("aria-current", "page");
  });
});
```

- [ ] **Step 5.7: Stub the new pages so the build succeeds**

```tsx
// study-plan/frontend/src/pages/HandbookPage.tsx
export function HandbookPage(): JSX.Element {
  return <p className="text-slate-500">Handbook coming next…</p>;
}
```

```tsx
// study-plan/frontend/src/pages/ResourcesPage.tsx
export function ResourcesPage(): JSX.Element {
  return <p className="text-slate-500">Resources coming next…</p>;
}
```

- [ ] **Step 5.8: Run vitest + tsc + build**

```bash
cd study-plan/frontend
npx tsc --noEmit
npm run test -- --run TabBar
npm run build
```

Expected: tsc clean; 4 TabBar tests pass; build succeeds.

- [ ] **Step 5.9: Commit**

```bash
git add study-plan/frontend/src
git commit -m "feat(frontend): router + TabBar + page shells"
```

---

## Task 6: Frontend — HandbookPage with markdown rendering

**Files:**
- Create: `study-plan/frontend/src/components/handbook/HandbookNav.tsx`
- Create: `study-plan/frontend/src/components/handbook/HandbookContent.tsx`
- Create: `study-plan/frontend/src/components/handbook/HandbookContent.test.tsx`
- Modify: `study-plan/frontend/src/pages/HandbookPage.tsx`

- [ ] **Step 6.1: Implement `HandbookNav.tsx`**

```tsx
import { NavLink } from "react-router-dom";
import type { HandbookChapterMeta } from "../../types";

interface Props {
  chapters: HandbookChapterMeta[];
}

export function HandbookNav({ chapters }: Props): JSX.Element {
  return (
    <nav aria-label="Handbook chapters" className="space-y-1">
      {chapters.map((c) => (
        <NavLink
          key={c.slug}
          to={`/handbook/${c.slug}`}
          className={({ isActive }) =>
            [
              "block rounded px-3 py-2 text-sm",
              isActive ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100",
            ].join(" ")
          }
        >
          <span className="font-medium">{c.title}</span>
          {c.subtitle ? <span className="block text-xs opacity-80">{c.subtitle}</span> : null}
        </NavLink>
      ))}
    </nav>
  );
}
```

- [ ] **Step 6.2: Implement `HandbookContent.tsx`**

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import type { HandbookChapter } from "../../types";

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
        rehypePlugins={[rehypeSlug, [rehypeAutolinkHeadings, { behavior: "wrap" }], rehypeHighlight]}
      >
        {chapter.body_md}
      </ReactMarkdown>
    </article>
  );
}
```

(Tailwind `prose` styling assumes `@tailwindcss/typography` is in the project.
If the migration plan didn't add it, install now:
`cd study-plan/frontend && npm install --save-dev @tailwindcss/typography`,
then add `require('@tailwindcss/typography')` to the `plugins` array in
`tailwind.config.js`.)

- [ ] **Step 6.3: Implement `HandbookPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { ApiClient } from "../api";
import { HandbookNav } from "../components/handbook/HandbookNav";
import { HandbookContent } from "../components/handbook/HandbookContent";
import type { HandbookChapter, HandbookChapterMeta } from "../types";

const api = new ApiClient(); // adapt if api.ts exports a singleton instead

export function HandbookPage(): JSX.Element {
  const { slug } = useParams();
  const [chapters, setChapters] = useState<HandbookChapterMeta[] | null>(null);
  const [chapter, setChapter] = useState<HandbookChapter | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listHandbookChapters().then(setChapters).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!slug) return;
    setChapter(null);
    api
      .getHandbookChapter(slug)
      .then((c) => setChapter(c))
      .catch((e) => setError(e.message));
  }, [slug]);

  if (error) return <p className="text-red-600">Error: {error}</p>;
  if (!chapters) return <p className="text-slate-500">Loading chapters…</p>;

  return (
    <div className="grid gap-6 md:grid-cols-[16rem_1fr]">
      <aside className="md:sticky md:top-6 md:self-start">
        <HandbookNav chapters={chapters} />
      </aside>
      <section>
        {chapter ? (
          <HandbookContent chapter={chapter} />
        ) : slug ? (
          <p className="text-slate-500">Loading chapter “{slug}”…</p>
        ) : (
          <p className="text-slate-500">Select a chapter.</p>
        )}
      </section>
    </div>
  );
}
```

- [ ] **Step 6.4: Write Vitest spec for `HandbookContent`**

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

- [ ] **Step 6.5: Run handbook tests + build**

```bash
cd study-plan/frontend
npx tsc --noEmit
npm run test -- --run HandbookContent
npm run build
```

Expected: tsc clean; 4 tests pass; build succeeds.

- [ ] **Step 6.6: Commit**

```bash
git add study-plan/frontend/src
git commit -m "feat(frontend): HandbookPage with markdown rendering"
```

---

## Task 7: Frontend — shared components (WeekBadge, EmptyState)

**Files:**
- Create: `study-plan/frontend/src/components/shared/WeekBadge.tsx`
- Create: `study-plan/frontend/src/components/shared/EmptyState.tsx`

These two are needed by ResourcesPage and may also be reused on Dashboard.

- [ ] **Step 7.1: Implement `WeekBadge.tsx`**

```tsx
interface Props {
  weeks: number[];
  highlightWeek?: number; // current week — render with accent style
}

export function WeekBadge({ weeks, highlightWeek }: Props): JSX.Element | null {
  if (!weeks.length) return null;
  const label = weeks.length === 1 ? `W${weeks[0]}` : `W${weeks[0]}–W${weeks[weeks.length - 1]}`;
  const active = highlightWeek !== undefined && weeks.includes(highlightWeek);
  return (
    <span
      className={[
        "inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium tabular-nums",
        active ? "bg-amber-100 text-amber-900" : "bg-slate-100 text-slate-700",
      ].join(" ")}
    >
      {label}
    </span>
  );
}
```

- [ ] **Step 7.2: Implement `EmptyState.tsx`**

```tsx
interface Props {
  title: string;
  description?: string;
}

export function EmptyState({ title, description }: Props): JSX.Element {
  return (
    <div
      role="status"
      className="rounded border border-dashed border-slate-300 bg-white px-6 py-10 text-center"
    >
      <h3 className="text-sm font-medium text-slate-700">{title}</h3>
      {description ? <p className="mt-1 text-xs text-slate-500">{description}</p> : null}
    </div>
  );
}
```

- [ ] **Step 7.3: Commit**

```bash
git add study-plan/frontend/src/components/shared
git commit -m "feat(frontend): shared WeekBadge + EmptyState components"
```

---

## Task 8: Frontend — ResourcesPage (filters + groups + cards)

**Files:**
- Create: `study-plan/frontend/src/components/resources/ResourceCard.tsx`
- Create: `study-plan/frontend/src/components/resources/ResourceGroup.tsx`
- Create: `study-plan/frontend/src/components/resources/ResourceFilters.tsx`
- Create: `study-plan/frontend/src/components/resources/ResourceFilters.test.tsx`
- Create: `study-plan/frontend/src/components/resources/ResourcesPage.test.tsx`
- Modify: `study-plan/frontend/src/pages/ResourcesPage.tsx`

- [ ] **Step 8.1: Implement `ResourceCard.tsx`**

```tsx
import type { ResourceItem } from "../../types";
import { WeekBadge } from "../shared/WeekBadge";

interface Props {
  resource: ResourceItem;
  currentWeek?: number;
}

export function ResourceCard({ resource, currentWeek }: Props): JSX.Element {
  const thisWeek =
    currentWeek !== undefined && resource.weeks.includes(currentWeek);
  const openable = resource.available && resource.href !== "";
  return (
    <article className="relative rounded border border-slate-200 bg-white p-4 shadow-sm">
      {thisWeek ? (
        <span className="absolute right-2 top-2 rounded bg-amber-500 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-white">
          This Week
        </span>
      ) : null}
      <h3 className="text-sm font-semibold text-slate-900">{resource.title}</h3>
      {resource.description ? (
        <p className="mt-1 text-xs text-slate-600">{resource.description}</p>
      ) : null}
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
        {resource.year ? <span className="tabular-nums">{resource.year}</span> : null}
        <WeekBadge weeks={resource.weeks} highlightWeek={currentWeek} />
        {resource.jd_tags.map((t) => (
          <span key={t} className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-700">
            {t}
          </span>
        ))}
      </div>
      <div className="mt-3">
        {openable ? (
          <a
            href={resource.href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs font-medium text-blue-600 hover:underline"
          >
            打开 →
          </a>
        ) : (
          <span
            className="text-xs text-slate-400"
            title="未下载，运行 bash resources/download_all.sh"
          >
            未下载
          </span>
        )}
      </div>
    </article>
  );
}
```

- [ ] **Step 8.2: Implement `ResourceGroup.tsx`**

```tsx
import type { ResourceItem } from "../../types";
import { ResourceCard } from "./ResourceCard";

const TYPE_LABELS: Record<string, string> = {
  paper: "Papers",
  repo: "Repositories",
  tutorial: "Tutorials",
  manual: "Manuals",
  blog: "Blogs",
};

interface Props {
  type: string;
  items: ResourceItem[];
  currentWeek?: number;
}

export function ResourceGroup({ type, items, currentWeek }: Props): JSX.Element | null {
  if (!items.length) return null;
  return (
    <section className="space-y-3">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600">
        {TYPE_LABELS[type] ?? type} <span className="text-slate-400">({items.length})</span>
      </h2>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {items.map((r) => (
          <ResourceCard key={r.id} resource={r} currentWeek={currentWeek} />
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 8.3: Implement `ResourceFilters.tsx`**

```tsx
import { useSearchParams } from "react-router-dom";
import type { ResourceFacets } from "../../types";

interface Props {
  facets: ResourceFacets;
}

export function ResourceFilters({ facets }: Props): JSX.Element {
  const [params, setParams] = useSearchParams();
  const set = (key: string, value: string | null) => {
    const next = new URLSearchParams(params);
    if (value === null || value === "" || value === "all") next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true });
  };

  const week = params.get("week") ?? "all";
  const tag = params.get("tag") ?? "all";
  const type = params.get("type") ?? "all";
  const q = params.get("q") ?? "";

  return (
    <div className="flex flex-wrap items-center gap-3">
      <input
        type="search"
        placeholder="Search title or description"
        value={q}
        onChange={(e) => set("q", e.target.value)}
        className="rounded border border-slate-300 px-3 py-1.5 text-sm"
        aria-label="Search resources"
      />
      <select value={week} onChange={(e) => set("week", e.target.value)} className="rounded border border-slate-300 px-2 py-1.5 text-sm" aria-label="Filter by week">
        <option value="all">All Weeks</option>
        {facets.weeks.map((w) => (
          <option key={w} value={String(w)}>{`W${w}`}</option>
        ))}
      </select>
      <select value={tag} onChange={(e) => set("tag", e.target.value)} className="rounded border border-slate-300 px-2 py-1.5 text-sm" aria-label="Filter by tag">
        <option value="all">All Tags</option>
        {facets.tags.map((t) => (
          <option key={t} value={t}>{t}</option>
        ))}
      </select>
      <div className="flex gap-1" role="tablist" aria-label="Filter by type">
        {["all", ...facets.types].map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={type === t}
            onClick={() => set("type", t)}
            className={[
              "rounded px-2 py-1 text-xs",
              type === t ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200",
            ].join(" ")}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 8.4: Implement `ResourcesPage.tsx`**

```tsx
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ApiClient } from "../api";
import type { ResourcesPayload } from "../types";
import { ResourceFilters } from "../components/resources/ResourceFilters";
import { ResourceGroup } from "../components/resources/ResourceGroup";
import { EmptyState } from "../components/shared/EmptyState";

const api = new ApiClient();

export function ResourcesPage(): JSX.Element {
  const [params] = useSearchParams();
  const [data, setData] = useState<ResourcesPayload | null>(null);
  const [currentWeek, setCurrentWeek] = useState<number | undefined>();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listResources().then(setData).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    api
      .getProgress()
      .then((p) => setCurrentWeek(p?.current_day?.week))
      .catch(() => {});
  }, []);

  const filtered = useMemo(() => {
    if (!data) return [];
    const week = params.get("week");
    const tag = params.get("tag");
    const type = params.get("type");
    const q = (params.get("q") ?? "").trim().toLowerCase();
    return data.resources.filter((r) => {
      if (type && type !== "all" && r.type !== type) return false;
      if (week && week !== "all" && !r.weeks.includes(Number(week))) return false;
      if (tag && tag !== "all" && !r.jd_tags.includes(tag)) return false;
      if (q && !`${r.title} ${r.description}`.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [data, params]);

  if (error) return <p className="text-red-600">Error: {error}</p>;
  if (!data) return <p className="text-slate-500">Loading…</p>;

  const groups = data.facets.types
    .map((t) => ({ type: t, items: filtered.filter((r) => r.type === t) }))
    .filter((g) => g.items.length > 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-slate-900">Resources</h2>
        {currentWeek !== undefined ? (
          <span className="rounded bg-amber-50 px-2 py-1 text-xs text-amber-900">
            本周 = W{currentWeek}
          </span>
        ) : null}
      </div>
      <ResourceFilters facets={data.facets} />
      {groups.length === 0 ? (
        <EmptyState title="No resources match the current filters." description="Adjust week / tag / type / search above." />
      ) : (
        <div className="space-y-8">
          {groups.map((g) => (
            <ResourceGroup key={g.type} type={g.type} items={g.items} currentWeek={currentWeek} />
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 8.5: Write Vitest spec for filters**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter, Routes, Route, useSearchParams } from "react-router-dom";
import { ResourceFilters } from "./ResourceFilters";

const facets = { types: ["paper", "repo"], weeks: [1, 2, 3], tags: ["kernel", "quant"] };

function Probe() {
  const [params] = useSearchParams();
  return <span data-testid="probe">{params.toString()}</span>;
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/resources" element={<><ResourceFilters facets={facets} /><Probe /></>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ResourceFilters", () => {
  it("writes week selection to query string", () => {
    renderAt("/resources");
    fireEvent.change(screen.getByLabelText("Filter by week"), { target: { value: "2" } });
    expect(screen.getByTestId("probe").textContent).toContain("week=2");
  });

  it("removes the param when 'all' is chosen", () => {
    renderAt("/resources?week=2");
    fireEvent.change(screen.getByLabelText("Filter by week"), { target: { value: "all" } });
    expect(screen.getByTestId("probe").textContent ?? "").not.toContain("week=");
  });

  it("clicking a type chip toggles type param", () => {
    renderAt("/resources");
    fireEvent.click(screen.getByRole("tab", { name: "paper" }));
    expect(screen.getByTestId("probe").textContent).toContain("type=paper");
  });
});
```

- [ ] **Step 8.6: Run vitest + tsc + build**

```bash
cd study-plan/frontend
npx tsc --noEmit
npm run test -- --run resources
npm run build
```

Expected: tsc clean; tests pass; build succeeds.

- [ ] **Step 8.7: Commit**

```bash
git add study-plan/frontend/src
git commit -m "feat(frontend): ResourcesPage with filters + groups + cards"
```

---

## Task 9: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 9.1: Run all backend tests**

```bash
pytest tests/test_study_plan_dashboard.py -v
```

Expected: all PASS (existing tests + ~16 new ones from Tasks 1 + 3).

- [ ] **Step 9.2: Run all frontend tests**

```bash
cd study-plan/frontend
npm run test -- --run
npx tsc --noEmit
npm run build
```

Expected: vitest green; tsc clean; build succeeds.

- [ ] **Step 9.3: Build the React bundle so the Python server can serve it**

The migration plan should already wire this; if so, `npm run build` writes
to `study-plan/static/`. Confirm:

```bash
ls study-plan/static/index.html
```

If this fails the migration plan was incomplete — stop and finish it
before continuing.

- [ ] **Step 9.4: Start the dashboard server and run the manual checklist**

```bash
python study-plan/dashboard.py --serve
# in another terminal:
curl -s http://localhost:8765/api/handbook | head -1
curl -s http://localhost:8765/api/resources | head -1
```

Then open `http://localhost:8765/` in a browser and walk through:

- [ ] Three tabs visible: **Dashboard**, **Handbook**, **Resources**
- [ ] Click **Handbook** → URL becomes `/handbook/sop` and Chapter 1 renders
- [ ] Click chapter 2/3/4 in left nav → URL updates, content swaps, no full reload
- [ ] Refresh on `/handbook/methodology` → still on Chapter 2
- [ ] Open a code block on any chapter — verify syntax highlighting CSS classes are applied
- [ ] Click an H2 heading — anchor URL `#...` appears in address bar
- [ ] Click **Resources** → groups visible (Papers ≥ count of local PDFs; Blogs from README)
- [ ] Type into the search box → list filters; URL updates with `?q=...`
- [ ] Pick a week from the dropdown → URL `?week=N`, list narrows
- [ ] Click a type chip → URL `?type=paper`, list narrows; click again to clear
- [ ] Click "在浏览器打开" on an available paper → PDF opens in new tab from `/resources-static/...`
- [ ] Pick a paper that's listed in `resources/README.md` but not present locally — verify "未下载" placeholder is shown and disabled
- [ ] Resize the window to ≤640px — three tabs and Handbook nav both stay usable

- [ ] **Step 9.5: Negative checks**

- [ ] Visit `/handbook/no-such-slug` → "Chapter not found" message, no crash
- [ ] Visit `/resources-static/../dashboard.py` → 404 (path traversal blocked)

- [ ] **Step 9.6: Final commit (only if any cleanup needed)**

If everything passes, no further commit. If you tweaked anything during manual checks (typo fixes, copy adjustments), commit them now:

```bash
git add -A
git commit -m "chore: handbook/resources manual-check fixes"
```

---

## Self-Review

Spec coverage check:

| Spec section | Implementing task |
|---|---|
| Handbook file contract (frontmatter, NN-slug.md) | Task 1 (loader + validation) + Task 2 (skeleton files) |
| `/api/handbook` + `/api/handbook/<slug>` | Task 1 |
| Resources auto-derived from filesystem | Task 3a |
| Resources README parsing (week, tag, blog) | Task 3b |
| `/api/resources` + facets | Task 3c |
| `/resources-static/` mount + traversal guard | Task 3c |
| Three top-level tabs / React Router | Task 5 |
| `Layout.tsx` + `TabBar` | Task 5 |
| `WeekBadge` + `EmptyState` shared | Task 7 |
| HandbookPage with `react-markdown` + plugins | Task 6 |
| ResourcesPage with filter URL state | Task 8 |
| Current-week chip via `getProgress()` | Task 8.4 |
| Type/Week/Tag/Search filters | Task 8.3 + 8.4 |
| Empty/error states | Tasks 6.3 + 8.4 (uses EmptyState) |
| Pytest backend tests (per-file in spec §5.1) | Tasks 1 + 3 |
| Vitest frontend tests (per spec §5.2) | Tasks 5 + 6 + 8 |
| Manual checklist (spec §5.3) | Task 9 |

No gaps detected.

Type consistency:
- `HandbookChapterMeta` / `HandbookChapter` (Task 4) used unchanged in Tasks 5–6.
- `ResourceItem` / `ResourceFacets` / `ResourcesPayload` (Task 4) used unchanged in Tasks 7–8.
- `ApiClient.listHandbookChapters/getHandbookChapter/listResources` (Task 4) consumed in Tasks 6.3 + 8.4 with matching signatures.
