# Handbook Design

Date: 2026-05-31 (revised after dashboard-react-shadcn merge)

## 历史与改名

本文档原名 *Handbook & Resources Design*，曾包含两个模块：

1. **Handbook**（训练指导流程书）—— 4 章 markdown 系统手册
2. **Resources**（参考资料栏目）—— 自动扫描 `resources/` 并解析 README

修订记录（2026-05-31）：发现 `dashboard-react-shadcn` 分支已经实现了 **References CRUD 视图**——
用户在前端手工增删改 `progress.yaml.references[]`（48+ 条已有条目，含 paper/blog/docs/video/repo/other 类目）。
References 视图的「人工策展精选清单」定位与本文档原 Resources 的「自动扫描 + 全量索引」定位**相互替代**，且 References 已经投入使用、内容已积累。

**结论**：取消 Resources 模块。本文档现在只覆盖 Handbook。

## 概述

为 study-plan 前端补一个新模块：

**Handbook（训练指导流程书）** —— 4 章 markdown 文件构成的系统手册：执行 SOP（怎么干）、方法论与原则（为什么）、工具手册（怎么用仓）、故障排除 + 达标样本（出事/合格）。

Handbook 作为 React 前端 sidebar 的**第 8 个 view** 加入（与现有的 focus / plan / references / operators / libraries / risks / tags 平级）。

## 前置依赖

`dashboard-react-shadcn` 分支已经合入 main。该分支提供：
- React + shadcn/ui 应用骨架（Vite 构建到 `study-plan/static/`）
- `Sidebar.tsx`（左侧 lg:w-52 / w-16 折叠 nav）
- `DashboardApp.tsx`（view 状态机驱动的内容区）
- `getDashboard()` API 客户端 + `/api/dashboard` 后端
- shadcn 原语（Button、Card、Select、ScrollArea 等）

执行计划在前置依赖完成前不能开始实施。

## 不在范围

- Handbook 章节正文的具体内容（本期只保证 4 个 md 文件存在、可被渲染、有最小骨架；详细内容是后续持续维护事务）
- markdown 内 admonition（`:::tip` 这类语法）—— 本期纯 GFM
- 章节全文搜索
- 章节内"上一章 / 下一章"翻页 UI（章数少，左侧鸡内导航更快）
- markdown 文件热重载（本地工具，重启 Python 进程）
- Handbook 内容的在线编辑（只读展示）
- 跨字段语义校验（例如 weeks 必须是连续整数）
- URL deep-link（如 `?chapter=sop`）—— 章数少、且 sidebar view 已经是非路由状态机，本期保持一致

## 整体架构

```text
docs/handbook/                       # 内容（markdown 4 章 + 资产）
  01-sop.md
  02-methodology.md
  03-tools.md
  04-troubleshooting.md
  assets/                            # 章节内引用的图（如有）

study-plan/dashboard.py              # 已合入分支后的形态 + 新增
  GET /api/handbook                  # 章节元数据列表（无 body）
  GET /api/handbook/<slug>           # 单章节，含 body_md
  （/api/dashboard 不变；不内联 handbook，避免每次 dashboard 刷新都拉 markdown）

study-plan/frontend/src/             # 新增 1 view + 2 组件
  components/dashboard/
    Sidebar.tsx                      # 修改：navItems 加 "handbook"
    DashboardApp.tsx                 # 修改：view==="handbook" 分支
  components/handbook/
    HandbookView.tsx                 # 顶层 view（拉数据 + 双栏布局）
    HandbookNav.tsx                  # 章节列表（左栏）
    HandbookContent.tsx              # 渲染 markdown（右栏）
```

注：原 spec 提的 `WeekBadge`、`EmptyState` 不再需要（Resources 不做了；分支已自带 `EmptyState`）。

## 数据契约

### Handbook 文件契约

**位置**：`docs/handbook/NN-slug.md`，其中 `NN` 为两位数字（章序），`slug` 是 URL-safe 的 kebab-case slug。

**frontmatter**（YAML，必填项粗体）：

```yaml
---
order: 1                # 必：整数，章序，决定渲染顺序
slug: sop               # 必：与文件名中的 slug 一致；用于 view-state 内部引用
title: 执行 SOP          # 必：UI 中的章节标题
subtitle: …             # 可选：标题下方的副标题（短句）
icon: play              # 可选：lucide-react 图标名；不填则不显示图标
---
```

**正文**：标准 GFM markdown。允许相对引用 `assets/` 下的图片。

**校验**（`load_handbook` 在加载时强制）：
- frontmatter 必须存在且能解析
- `order` / `slug` / `title` 必填
- `slug` 必须等于文件名中提取的 slug（防止重命名失同步）
- 同一目录内 `slug` 不重复
- 同一目录内 `order` 不重复

任何一条不通过 → 抛 `HandbookError`（含具体路径与原因）。**主进程在启动时不调用 `load_handbook`，仅在请求 `/api/handbook*` 时按需调用**——这样章节文件的写错不影响 dashboard 启动。

### Handbook API

**`GET /api/handbook`** — 仅元数据，按 `order` 升序：

```json
{
  "chapters": [
    { "order": 1, "slug": "sop", "title": "执行 SOP", "subtitle": "…", "icon": "play" },
    { "order": 2, "slug": "methodology", "title": "方法论与原则", "subtitle": null, "icon": null },
    …
  ]
}
```

**`GET /api/handbook/<slug>`** —  单章节，含正文：

```json
{
  "order": 1,
  "slug": "sop",
  "title": "执行 SOP",
  "subtitle": "…",
  "icon": "play",
  "body_md": "# 执行 SOP\n\n## 一天的闭环\n…"
}
```

未知 slug → `404 {"error": "not found"}`。

### Frontend 类型

`study-plan/frontend/src/types.ts` 增量：

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

`study-plan/frontend/src/api.ts` 增量（沿用分支的 fetch-helper 风格）：

```ts
export async function listHandbookChapters(): Promise<HandbookChapterMeta[]>;
export async function getHandbookChapter(slug: string): Promise<HandbookChapter | null>;
```

## UI 设计

### Sidebar 集成

```tsx
// Sidebar.tsx 改动
export type View =
  | "focus" | "plan" | "operators" | "libraries"
  | "risks" | "tags" | "references" | "handbook";   // ← 新增

const navItems = [
  // …已有 7 项…
  { id: "handbook", label: "Handbook", icon: <BookText className="h-5 w-5" /> },
];
```

`BookText` 来自 `lucide-react`，是 References (`BookOpen`) 之外的另一个书籍图标，区分两类。

### HandbookView 双栏布局

```text
┌─────────────────────────────────────────────────────────┐
│ Sidebar │ HandbookView (in <main>)                      │
│         │ ┌───────────┬───────────────────────────────┐ │
│ Focus   │ │ HandbookNav│ HandbookContent              │ │
│ Plan    │ │ (16rem)   │ ┌───────────────────────────┐ │ │
│ Refs    │ │ • SOP     │ │ # 执行 SOP                │ │ │
│ Ops     │ │ • Method… │ │                           │ │ │
│ Libs    │ │ • Tools   │ │ ## 一天的闭环             │ │ │
│ Risks   │ │ • Trouble…│ │ 1. python …               │ │ │
│ Tags    │ │           │ │ …                         │ │ │
│*Handbook│ │           │ │                           │ │ │
│         │ └───────────┴───────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

- HandbookNav 用同样的 shadcn `Card` 容器风格，与现有 sidebar/卡片视觉一致
- 选中章节用蓝色高亮（与 Sidebar 选中态相同 `bg-blue-100 text-blue-700`）
- 状态保存在 `HandbookView` 内 `useState<string>`(默认 "sop")，**不进 URL**——保持与 sidebar view 一致的非路由模型
- 移动端折叠：`md:` 之下用 shadcn `Select` 当章节切换器（横向占满），章节切换后渲染内容

### HandbookContent 渲染管线

`react-markdown` 配以下插件（保持与原 spec 一致）：

- `remark-gfm` —— 表格、任务列表、删除线
- `rehype-slug` + `rehype-autolink-headings` —— H1-H4 自动加 anchor `id`，可在地址栏复制 #fragment
- `rehype-highlight` —— 代码块语法高亮，主题使用 `highlight.js/styles/github.css`

样式：Tailwind `prose prose-slate max-w-none`（要求 `@tailwindcss/typography` 已安装；如果迁移分支没装，本计划 Step 1 装）。

### 加载与错误状态

- 章节列表加载中 → 复用分支的 `<LoadingState />`
- 章节内容加载中 → 单行 `<p className="text-slate-500">Loading…</p>`
- 章节列表请求失败 → 复用分支的 `<EmptyState title="Unable to load handbook" role="alert">…</EmptyState>`
- 单章节 404 → 内容区显示 `<EmptyState title="Chapter not found">` + 提示选择左侧其他章节

## 后端实现关键点

### `load_handbook()` 行为

返回 `list[dict[str, Any]]`，每条含全部 frontmatter 字段 + `body_md`。`order` 升序排序。

```python
def load_handbook() -> list[dict[str, Any]]:
    if not HANDBOOK_DIR.exists():
        return []
    chapters = []
    seen_slugs: set[str] = set()
    seen_orders: set[int] = set()
    for path in sorted(HANDBOOK_DIR.glob("[0-9][0-9]-*.md")):
        # parse frontmatter; validate order/slug/title; ensure no duplicates
        ...
    return sorted(chapters, key=lambda c: c["order"])
```

### Endpoint dispatch

在 `DashboardHandler.do_GET`（合并后的形态）的 API 分支中，**`/api/dashboard` 之前**或**之后**插入：

```python
if parsed.path == "/api/handbook":
    json_response(self, api_handbook_index())
    return
if parsed.path.startswith("/api/handbook/"):
    slug = parsed.path[len("/api/handbook/"):]
    if not slug or "/" in slug:
        self.send_error(404)
        return
    chapter = api_handbook_chapter(slug)
    if chapter is None:
        json_response(self, {"error": "not found"}, code=404)
    else:
        json_response(self, chapter)
    return
```

不需要静态 mount（原 Resources 才需要 `/resources-static/`，已经取消）。

## 测试计划

### Backend (`pytest tests/test_study_plan_dashboard.py`)

- `test_load_handbook_returns_chapters_in_order` —— 多文件按 order 排序
- `test_load_handbook_rejects_filename_slug_mismatch` —— 文件名与 frontmatter slug 不一致 → `HandbookError`
- `test_load_handbook_rejects_missing_required_field` —— 缺 order / slug / title → `HandbookError`
- `test_load_handbook_rejects_duplicate_slug` —— 两个文件同 slug → `HandbookError`
- `test_load_handbook_rejects_duplicate_order` —— 两个文件同 order → `HandbookError`
- `test_load_handbook_returns_empty_when_dir_missing` —— 目录不存在 → `[]`，不抛
- `test_load_handbook_keeps_optional_fields_none_when_absent` —— subtitle / icon 缺失 → `None`
- `test_api_handbook_lists_chapters_meta_only` —— `/api/handbook` 不含 `body_md`
- `test_api_handbook_chapter_returns_body` —— `/api/handbook/<slug>` 返回完整章节
- `test_api_handbook_chapter_unknown_slug_returns_none` —— 未知 slug → 404

### Frontend (`vitest`)

- `HandbookContent.test.tsx` —— 渲染 title/subtitle、GFM 表格、代码高亮、heading anchors
- `HandbookNav.test.tsx` —— 列出所有章节、点击触发 onSelect、当前选中态高亮
- `HandbookView.test.tsx`（轻量）—— 拉数据 + 默认选 "sop" + 切换章节后内容更新

### 手动验收（merge 之后跑）

- 启动 `python study-plan/dashboard.py --serve`，浏览器打开
- 侧边栏出现 **Handbook** 第 8 项（图标 BookText）
- 点击 Handbook → 默认渲染第 1 章（SOP），左栏列出 4 章
- 点 Methodology / Tools / Troubleshooting，右栏内容切换；URL 不变（设计如此）
- H2 标题点击 → 地址栏出现 `#…` fragment
- 代码块有语法高亮
- 任意 GFM 表格能渲染
- 移动端 (≤640px) sidebar 折叠为 16px，HandbookNav 改为 Select；切换可用
- 制造一个错误：把 `01-sop.md` 的 slug 改成 `sopx`，刷新前端 → Handbook 区域显示 EmptyState「Unable to load handbook」；其他 view 不受影响（因为 `load_handbook` 只在请求时调）

## 与现有 References 的区分

| 维度 | References (已存在) | Handbook (本计划) |
|---|---|---|
| 内容来源 | 用户在 UI 手工录入，存 progress.yaml | 文件系统下 markdown 文件 |
| 写入路径 | UI CRUD → POST /api/references | 编辑器写入 docs/handbook/*.md，重启或刷新可见 |
| 数据模型 | 平铺 array of {id,title,url,category,notes} | 章节有顺序、frontmatter、长正文 |
| 适用场景 | 学习中追加的零散链接、论文 | 系统级、稳定的方法论 / SOP / 工具 |
| Sidebar 入口 | `references` (BookOpen) | `handbook` (BookText) |

两者并存、互不取代。Handbook 是「读」(规则、流程)，References 是「写」(收藏、记录)。

## 风险与回滚

- **风险**：HandbookContent 引入 6 个新 npm 依赖（react-markdown 全家），增加 bundle 体积约 ~80KB gzipped。仅当用户切到 handbook view 才需要——可后续考虑 React.lazy 切片，但本期不做。
- **风险**：highlight.js 默认 bundle 包含全语言；首次加载稍重。本期接受；如优化只引入 `python/bash/typescript` 三种。
- **回滚**：单 sidebar item + 一个新文件夹组件，即使整体回滚也只需还原 Sidebar.tsx + DashboardApp.tsx + 删除 `components/handbook/` + 后端两条路由分支。

## 自检

- [x] 所有原文档涉及 Resources 的章节都标注「移除」或迁移到「不在范围」
- [x] 数据契约只剩 Handbook
- [x] 测试覆盖与新范围匹配（去掉 Resources 测试，保留 Handbook 全部）
- [x] UI 设计跟分支已有的 Sidebar / DashboardApp 形态对齐（不再引入 react-router-dom / TabBar）
- [x] 与已有 References 的边界写清楚
- [x] 描述了 frontmatter 校验失败的隔离策略（按需加载，不影响主进程启动）
