# Handbook & Resources Design

Date: 2026-05-31

## 概述

为 study-plan 前端补两个新模块：

1. **Handbook（训练指导流程书）** — 4 章 markdown 文件构成的系统手册：执行 SOP（怎么干）、方法论与原则（为什么）、工具手册（怎么用仓）、故障排除 + 达标样本（出事/合格）。
2. **Resources（参考资料栏目）** — 自动扫描 `resources/` 目录、解析 `resources/README.md` 元数据，按 type 分组 + 周/JD tag/搜索过滤展示所有学习参考资料。

两个模块作为 React 前端的**独立顶级 Tab** 加入：`/`（Dashboard）、`/handbook`、`/resources`。

## 前置依赖

本设计依赖 `docs/superpowers/plans/2026-05-30-dashboard-react-shadcn.md` 已落地。React 应用骨架、Vite 构建、`Layout` 组件、`/api/progress` 客户端、Tailwind + shadcn/ui 已就位。本设计不重定义这些。

执行计划在前置依赖完成前可以先写、不能开始实施。

## 不在范围

- Handbook 章节正文的具体内容（本期只保证 4 个 md 文件存在、可被渲染、有最小骨架；详细内容是后续持续维护事务）
- Resource 加注释/评分功能
- markdown 内 admonition（`:::tip` 这类语法）—— 本期纯 GFM
- 章节全文搜索
- 章节内"上一章 / 下一章"翻页 UI（章数少，直接点左侧导航更快）
- markdown 文件热重载（本地工具，重启 Python 进程）
- Handbook 内容的在线编辑（只读展示）
- 跨字段语义校验（例如 weeks 必须是连续整数）

## 整体架构

```text
docs/handbook/                       # 内容（markdown 4 章 + 资产）
  01-sop.md
  02-methodology.md
  03-tools.md
  04-troubleshooting.md
  assets/                            # 章节内引用的图（如有）

study-plan/dashboard.py              # 新增 2 组只读 endpoint
  GET /api/handbook
  GET /api/handbook/<slug>
  GET /api/resources
  GET /resources-static/...           # 静态文件 mount（PDF 浏览）

study-plan/frontend/src/             # 新增路由 + 2 个页面 + 共享组件
  routes.tsx
  pages/HandbookPage.tsx
  pages/ResourcesPage.tsx
  components/handbook/
    HandbookNav.tsx
    HandbookContent.tsx
  components/resources/
    ResourceFilters.tsx
    ResourceGroup.tsx
    ResourceCard.tsx
  components/shared/
    WeekBadge.tsx
    EmptyState.tsx
  components/layout/
    TabBar.tsx
```

数据流：

- Handbook：markdown 文件原样保留 → Python 启动时 glob + frontmatter 解析 → `/api/handbook` 返回元数据列表 + 单章内容 → 前端 `react-markdown` 渲染。
- Resources：Python 启动时扫 `resources/papers|repos|tutorials|manuals/` + 解析 `resources/README.md` 提取 tag/week → `/api/resources` 返回结构化列表 + facets → 前端按 type/week/tag/搜索过滤。

单一职责边界：

- `dashboard.py` 仍只做"读 progress + 写 progress + 静态文件 + API"。新增的 2 个 endpoint 是只读的，不写任何文件。
- React app 的 `Layout` 加一个三段 TabBar；现有 Dashboard 内容退到 `/`，Handbook/Resources 各占新路由。
- markdown 文件本身就是手册（`docs/handbook/*.md`），任何文本编辑器都能改、git diff 友好、不依赖前端运行。

## Handbook 数据与渲染

### 文件契约

每章一个 markdown 文件，文件名固定 `NN-slug.md`（NN 决定排序），文件头是 yaml frontmatter：

```markdown
---
order: 1
slug: sop
title: 执行 SOP
subtitle: 一天/一周/一阶段的闭环动作
icon: play           # lucide-react 图标名（可选）
---

# 执行 SOP

## 一天的闭环
...
```

为什么 frontmatter 而非纯文件名：title/subtitle/icon 是 UI 元数据，不应该被塞进文件名；frontmatter 让作者改标题不用改路由 slug。`order` 显式给出而不是从文件名推导，保留以后插章的灵活度。

约束：

- 文件名 `NN-<slug>.md`，NN 两位数；slug 必须与 frontmatter 一致（启动时校验，不一致 → 启动失败、报清晰错误）
- 必填字段：`order`、`slug`、`title`
- 可选字段：`subtitle`、`icon`
- 内容用普通 GitHub-flavored markdown，支持代码块（语法高亮在前端）、表格、引用、图片相对路径

### Python 端

`study-plan/dashboard.py` 新增（追加，不重写已有逻辑）：

```python
HANDBOOK_DIR = BASE_DIR.parent / "docs" / "handbook"

def load_handbook() -> list[dict]:
    """启动时调用一次。返回按 order 排序的章节元数据 + 正文。
    解析失败抛异常 (启动失败胜过运行时坏数据)。"""
```

路由：

```text
GET /api/handbook
→ {"chapters": [{order, slug, title, subtitle, icon}, ...]}

GET /api/handbook/<slug>
→ {slug, title, subtitle, icon, body_md, order}
```

`load_handbook()` 在进程启动时跑一次，缓存结果。本地工具，无需热重载——改 markdown 后重启 Python 进程即可。

为什么不在前端 fetch markdown 文件本身：经过 Python 集中校验 + frontmatter 解析后，前端不用引入 yaml 解析器；同时为后续给章节加 cross-reference 留口子。

### 前端组件

```
HandbookPage.tsx
├── HandbookNav.tsx     左侧 sticky 章节列表（Card 内含 Button list）
└── HandbookContent.tsx 右侧渲染当前章节
    └── 用 react-markdown + remark-gfm + rehype-highlight
        + rehype-slug + rehype-autolink-headings
        ↳ 标题自动生成锚点；右上角小目录从 H2/H3 自动抽
```

URL 设计：`/handbook` 重定向到第一章 → `/handbook/sop`。每章 H2 锚点形如 `/handbook/sop#一天的闭环`。

为什么这套 markdown 渲染栈：

- `react-markdown + remark-gfm`：表格 / 任务列表 / `~~strikethrough~~` 标配。
- `rehype-highlight`：代码块语法高亮，纯 CSS，不拖慢 build（vs prismjs/shiki 体积更大）。
- `rehype-slug + rehype-autolink-headings`：稳定可分享锚点。

布局（桌面）：

```
┌─ Tabs: Dashboard | Handbook ● | Resources ─────────┐
├──────────────┬─────────────────────────────────────┤
│ ① 执行 SOP    │ # 执行 SOP                          │
│ ② 方法论原则  │ ## 一天的闭环                       │
│ ③ 工具手册    │ ...                                 │
│ ④ 故障排除   │                                     │
│              │                                     │
│ [TOC]        │                                     │
│ - 一天的闭环  │                                     │
│ - 一周的闭环  │                                     │
└──────────────┴─────────────────────────────────────┘
```

窄屏：左侧导航塌成顶部 `<Select>` 章节切换器，TOC 放抽屉。

### 内容初版边界

本 spec 落地时**先建 4 个 markdown 文件**，每个带：

- frontmatter 完整
- 至少一个 H2 + 一段示例内容（不要求一次性写满）

详细内容不在本次 spec 范围 —— 内容是后续持续维护的事，不应该跟前端框架绑死。本次只保证：4 个文件存在 + 可被渲染 + 有最小可用骨架。

## Resources 扫描与渲染

### 数据来源（自动派生）

Python 启动时扫描 `resources/`，按 type 分类：

| Type | 来源 | 标识方式 |
|------|------|----------|
| `paper` | `resources/papers/*.pdf` | 文件名前缀编号 `01_..._2018.pdf` 解析出 number/year/slug |
| `repo` | `resources/repos/*` 子目录（git clone） | 子目录名 = `<owner>_<name>` 或 `.git/config` 读 remote |
| `tutorial` | `resources/tutorials/*` | 子目录或单 html/md |
| `manual` | `resources/manuals/*` | 同上 |
| `blog` | 仅 `resources/README.md` 的"博客 / 技术文章"表格 | 无文件，纯链接 |

设计要在「目录为空」「文件已下载」「README 表里写了但本地没下载」三种状态下都能正确显示。

### 元数据来源 = `resources/README.md`

不另开 `resources.yaml`。README 现有的表格本身就是元数据源 —— 把它解析成结构化数据：

- **Week 范围**：从 `## 论文清单（按周排列）` 下的 `### Week N-M` 子标题取
- **JD tag**：从「用途」列里映射（启动时定一份手工 mapping，比如「FlashAttention」→ `kernel`、「vLLM」→ `framework, serving`），mapping 表写在 `dashboard.py` 一个常量字典里
- **描述**：「用途」列原文

边界状态：

- 当本地 PDF 存在 README 没列时：仍然展示，`type=paper`，`weeks=[]`，`jd_tags=[]`，description=`(no metadata)`
- 当 README 列了但本地不存在时：标 `available=false` 灰显

### 解析实现

```python
RESOURCES_DIR = BASE_DIR.parent / "resources"

@dataclass
class Resource:
    id: str            # stable: "paper:03_flashattention_dao_2022"
    type: Literal["paper","repo","tutorial","manual","blog"]
    title: str
    description: str
    weeks: list[int]   # [3, 4]
    jd_tags: list[str] # ["kernel"]
    href: str          # 本地 file path 或 https url
    available: bool    # 本地存在
    year: int | None

def load_resources() -> list[Resource]:
    """扫目录 + 解析 README → 合并 → 排序。启动时跑一次缓存。"""
```

启动时调用一次，缓存结果。改 README 或下载新文件 → 重启 Python 进程。

为什么不做实时 watch：本地工具，每天启动一次，不值得引入 watchdog。

### API

```
GET /api/resources
→ {
    "resources": [Resource, ...],
    "facets": {
      "types": ["paper","repo","tutorial","manual","blog"],
      "weeks": [1,2,...,8],
      "tags":  ["kernel","framework","serving","perf","quant","docs","interview"]
    }
  }
```

facets 与现有 `/api/progress` 的 `options.tags` 保持同名，前端 filter UI 复用同一组 tag 颜色 / 图标。

### 前端组件

```
ResourcesPage.tsx
├── ResourceFilters       搜索框 + Week Select + Tag Select + Type 切换 chips
├── ResourceGroup × N     按 type 分组，每组一个 Card
│   └── ResourceCard      title + meta（year, weeks, tags） + 操作按钮
└── 空状态                  无匹配 → EmptyState 并保留过滤条件
```

ResourceCard 内的操作：

- `type=paper`：「在浏览器打开」（`<a href="/resources-static/papers/03_..pdf" target="_blank">`，Python 端把 `resources/` mount 成 `/resources-static/`）
- `type=repo / tutorial / blog`：跳外链
- 当 `available=false`：禁用「打开」，提示「未下载，运行 `bash resources/download_all.sh`」

与当前进度联动（轻量，不喧宾夺主）：

- ResourcesPage 在初次加载时**额外**调一次 `getProgress()`（迁移 spec 已规划的 client 方法）取 `current_day.week`；只取这一个字段，失败时静默降级（不阻塞 Resources 渲染）。
- 顶部展示 chip：`本周 = W3` + 一键过滤「只看本周」。
- 当 `weeks` 包含当前周数时，ResourceCard 右上加一个「This Week」小徽标。

### 静态文件 mount

```python
# 把 PDF / 本地静态资料暴露为只读
self.send_file_in("/resources-static/", RESOURCES_DIR)
```

只允许 GET、限制路径在 `RESOURCES_DIR` 下（用 `os.path.commonpath` 校验防穿越）。

## 路由 · Layout · 共享组件

### 路由方案

引入 **React Router v6** 进 React 前端。React 迁移 spec 当时刻意排除了路由（"Vite is sufficient for this local tool; Next.js would add routing"），但那是说不需要服务器端路由 —— 客户端三 tab 路由是另一回事，必要且轻量（约 3 KB gzip）。

```text
/                 → DashboardPage   （即原 dashboard 内容，迁移 spec 已规划）
/handbook         → HandbookPage    重定向到第一章 /handbook/sop
/handbook/:slug   → HandbookPage 渲染对应章节
/resources        → ResourcesPage
/resources?week=3&tag=kernel&type=paper&q=...   过滤态写到 query string
```

为什么过滤写 URL：可分享、可后退、刷新不丢状态。用 `useSearchParams`，无需自管 state。

### Layout 改动

`Layout.tsx`（迁移 spec 已存在）顶部新增 `TabBar`：

```
┌─ LLM Kernel Lab ─────────────── Refresh ─┐
│ [Dashboard] [Handbook] [Resources]       │   ← 新增
├──────────────────────────────────────────┤
│  <Outlet />                              │
└──────────────────────────────────────────┘
```

TabBar 用 shadcn `Tabs` 的视觉，但行为走 React Router 的 `<NavLink>`（不是 client-only 的 Tab state），保证刷新后仍在当前页。

当前页高亮：`<NavLink>` 的 `aria-current="page"`，CSS 走 Tailwind data attribute。

### 共享组件

```
study-plan/frontend/src/components/
  ui/                           # shadcn primitives，迁移 spec 已带
    Tabs.tsx, Card.tsx, Badge.tsx, Button.tsx, Input.tsx, Select.tsx
  shared/
    TagBadge.tsx                # 已规划：JD tag 渲染（kernel/framework/...）
    WeekBadge.tsx               # 新：W1-W8 一致渲染
    EmptyState.tsx              # 新：空列表占位
  layout/
    Layout.tsx                  # 改：加 TabBar
    TabBar.tsx                  # 新
```

`TagBadge` 与 `WeekBadge` 在 Dashboard、Handbook（章节内引用）、Resources 三页共用同一组颜色与样式 —— 保证「kernel」标签长得一致，避免视觉碎片。

### API 客户端扩展

迁移 spec 里 `ApiClient` 包了 `/api/progress` 等。本次扩 2 组方法：

```typescript
class ApiClient {
  // 已有：getProgress, postDay, postOperator, postLibrary

  async listHandbookChapters(): Promise<HandbookChapterMeta[]>
  async getHandbookChapter(slug: string): Promise<HandbookChapter>
  async listResources(): Promise<{ resources: Resource[]; facets: Facets }>
}
```

错误处理：复用迁移 spec 已规划的 error boundary + toast 模式。Handbook 章节 404 时显示「章节不存在 → 回 Handbook 首页」，不崩页面。

### 数据加载策略

- **Handbook**：进入 `/handbook` 时 `listHandbookChapters` 一次拿到 4 章元数据；`getHandbookChapter(slug)` 在切章时按需请求，单章正文加载完成后缓存到组件 state，再切回不重发。章数少（4）、payload 小，懒加载已足够，不做预取。
- **Resources**：进入 `/resources` 时 `listResources` 一次。过滤纯前端，无需重新请求。
- **首屏 Dashboard 不变**：不为 Handbook/Resources 提前加载，避免主页变慢。

## 测试与验收

### 后端测试（pytest，加到 `tests/test_study_plan_dashboard.py`）

Handbook：

- `test_load_handbook_returns_chapters_in_order` — tmp 目录放 `02-x.md` `01-y.md`，按 `order` 升序返回
- `test_load_handbook_parses_frontmatter` — title/subtitle/icon 正确解析；body_md 不含 frontmatter
- `test_load_handbook_rejects_filename_slug_mismatch` — `01-foo.md` 里 frontmatter `slug: bar` → 启动失败
- `test_load_handbook_missing_required_fields` — 缺 `order` / `slug` / `title` → 启动失败、错误信息包含文件名
- `test_get_handbook_chapter_404_unknown_slug` — `/api/handbook/does-not-exist` → 404

Resources：

- `test_load_resources_scans_papers_dir` — tmp 放 3 个 PDF，返回 3 条 `type=paper`
- `test_load_resources_marks_unavailable_when_listed_in_readme_only` — README 列了但本地没文件 → `available=False`
- `test_load_resources_parses_week_ranges_from_readme` — `### Week 3-4` → `weeks=[3,4]`
- `test_load_resources_extracts_jd_tags_via_mapping` — 「FlashAttention」描述 → `tags 含 "kernel"`
- `test_load_resources_handles_empty_subdirs` — repos/tutorials/manuals 为空 → 各组返回 `[]`，不抛
- `test_resources_static_path_traversal_blocked` — 请求 `/resources-static/../dashboard.py` → 400/403

### 前端测试（Vitest + Testing Library）

Handbook：

- `HandbookNav` 渲染 4 章 + 当前章高亮
- `HandbookContent` 渲染 markdown：表格、代码块带语言 class、H2 自动锚点
- 切章节 URL 更新到 `/handbook/<slug>`
- 章节加载失败 → 显示错误提示，不渲染空白

Resources：

- 初次渲染：按 type 分组 + 计数正确
- 改 Week filter → URL `?week=3` 同步、列表过滤一致
- 搜索 "flash" → 命中 FlashAttention 论文
- `available=false` 的卡片：打开按钮 disabled，提示文案显示
- 空状态：无匹配时显示 EmptyState 并保留 filter

Layout / 路由：

- 三 tab 切换：URL 更新、`aria-current` 切换
- 直接访问 `/handbook/sop` 刷新页面 → 仍命中正确章节

### 手测验收 checklist

- [ ] `python study-plan/dashboard.py --serve` 启动后能访问 `/`、`/handbook`、`/resources`
- [ ] 4 章 markdown 在 Handbook 都能渲染，代码块有高亮，标题有锚点可分享
- [ ] Resources 至少能列出当前 22 篇 papers + README 表里所有外链 blogs
- [ ] 把 `resources/papers/` 下任意 PDF 临时改名 → Resources 显示该项 `not_downloaded`
- [ ] 改 markdown 文件、重启进程 → 内容更新；frontmatter 写错 → 启动失败带行号
- [ ] 三 tab 在窄屏（≤640 px）布局不崩

### 不在测试范围

- 内容质量本身（markdown 写得对不对是后续维护）
- PDF 渲染（交给浏览器内置 PDF viewer）
- 性能基准（数据集小、本地工具）

## 落地步骤摘要

1. **前置**：React 迁移 plan `2026-05-30-dashboard-react-shadcn.md` 完成
2. **后端**：`dashboard.py` 加 `load_handbook` / `load_resources` / 2 个 endpoint / `/resources-static` mount，配套 pytest
3. **内容初版**：`docs/handbook/01..04-*.md` 4 个文件最小骨架（每章 1 个 H2 + 几行内容）
4. **前端依赖**：装 `react-router-dom`、`react-markdown`、`remark-gfm`、`rehype-highlight`、`rehype-slug`、`rehype-autolink-headings`；引入 highlight.js 主题 CSS
5. **前端**：路由 + Layout TabBar + HandbookPage/HandbookNav/HandbookContent + ResourcesPage/Filters/Group/Card + 共享 WeekBadge/EmptyState
6. **测试**：pytest + vitest 全绿
7. **手测**：上节 checklist 全过

详细任务拆分留给随后的 implementation plan。
