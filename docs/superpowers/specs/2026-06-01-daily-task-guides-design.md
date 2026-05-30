# 每日任务详细引导系统设计

## 概述

为 study-plan 的 56 天计划中每个 task 和 artifact 添加结构化引导信息，包括操作步骤、完成标准、参考资料、时间估算和依赖关系。引导数据存储在独立 YAML 文件中，通过 Python API 合并后在 CurrentFocusPanel 中展示。

## 数据存储

### 文件位置

```
study-plan/guides/
├── day01.yaml
├── day02.yaml
├── ...
└── day56.yaml
```

每天一个文件，与 `progress.yaml` 分离，保持进度数据简洁。

### 文件结构

```yaml
day: 1

tasks:
  <task_key>:
    summary: 一句话描述这个 task 做什么
    steps:
      - 第一步操作描述
      - 第二步操作描述
      - ...
    done_when: 明确的完成标准，可验证
    time_minutes: 30
    depends_on: []          # 同一天内的前置 task key 列表
    refs:                   # 可选
      - title: 参考资料标题
        url: 链接或相对路径

artifacts:
  <artifact_key>:
    summary: 产出物描述
    done_when: 完成标准
    time_minutes: 15
    depends_on: [task_key]  # 依赖的 task

total_time_minutes: 105     # 所有 task + artifact 的时间总和
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `summary` | string | 是 | 一句话描述 |
| `steps` | string[] | 是 | 有序操作步骤 |
| `done_when` | string | 是 | 可验证的完成标准 |
| `time_minutes` | int | 是 | 预估耗时（分钟） |
| `depends_on` | string[] | 否 | 同天内前置 task key，默认 `[]` |
| `refs` | {title, url}[] | 否 | 参考资料列表 |

### 约束

- `tasks` 和 `artifacts` 的 key 必须与 `progress.yaml` 中对应 day 的 key 一致
- `depends_on` 只引用同一天内的 task/artifact key
- `total_time_minutes` 等于所有 task + artifact 的 `time_minutes` 之和

## API 变更

### `get_api_data()` 修改

加载当天对应的 guide 文件（如果存在），合并到 day 数据中：

```python
GUIDES_DIR = BASE_DIR / "guides"

def load_guide(day_num: int) -> dict | None:
    guide_file = GUIDES_DIR / f"day{day_num:02d}.yaml"
    if guide_file.exists():
        with open(guide_file, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return None
```

在 `enrich_day()` 中合并：

```python
def enrich_day(day: dict) -> dict:
    enriched = dict(day)
    # ... 现有逻辑 ...
    guide = load_guide(enriched["num"])
    if guide:
        enriched["guide"] = guide
    return enriched
```

### API 响应变化

`/api/progress` 中每个 day 对象新增可选 `guide` 字段：

```json
{
  "num": 1,
  "title": "仓库校准 + Nsight Compute WSL2 验证",
  "status": "not_started",
  "tasks": { "audit_existing_kernels": false },
  "guide": {
    "tasks": {
      "audit_existing_kernels": {
        "summary": "审计仓库中已有的 kernel 实现",
        "steps": ["列出 kernels/ 目录...", "检查每个 kernel..."],
        "done_when": "docs/audit.md 存在且...",
        "time_minutes": 30,
        "depends_on": [],
        "refs": [{"title": "...", "url": "..."}]
      }
    },
    "artifacts": { ... },
    "total_time_minutes": 105
  }
}
```

## 前端展示

### TypeScript 类型

```typescript
interface TaskGuide {
  summary: string;
  steps: string[];
  done_when: string;
  time_minutes: number;
  depends_on: string[];
  refs?: { title: string; url: string }[];
}

interface ArtifactGuide {
  summary: string;
  done_when: string;
  time_minutes: number;
  depends_on: string[];
  refs?: { title: string; url: string }[];
}

interface DayGuide {
  tasks: Record<string, TaskGuide>;
  artifacts: Record<string, ArtifactGuide>;
  total_time_minutes: number;
}
```

`DashboardDay` 类型新增可选字段：`guide?: DayGuide`

### CurrentFocusPanel 展示

在现有 Checklist 组件基础上增强，每个 task 项展开显示：

1. **Summary** — task 名称下方的灰色描述文字
2. **Steps** — 有序列表，带序号
3. **Done When** — 绿色边框的完成标准卡片
4. **Time** — 右侧显示预估时间 badge（如 "~30min"）
5. **Dependencies** — 如果有未完成的前置 task，显示黄色提示
6. **Refs** — 参考链接列表，可点击

### 布局

```
┌─ Task: Audit Existing Kernels ──────────── ~30min ─┐
│ ✓/○ 审计仓库中已有的 kernel 实现                      │
│                                                      │
│ Steps:                                               │
│  1. 列出 kernels/ 目录下所有 .py 和 .cu 文件          │
│  2. 检查每个 kernel 是否有对应的 test 文件             │
│  3. 记录缺失 test 的 kernel 到 docs/audit.md          │
│                                                      │
│ ✅ Done when: docs/audit.md 存在且列出了所有...        │
│                                                      │
│ 📎 项目 kernels 目录                                  │
└──────────────────────────────────────────────────────┘
```

### 无引导文件时的降级

如果某天没有对应的 guide 文件，CurrentFocusPanel 保持现有行为（只显示 task 名称和勾选状态），不报错。

## 内容生成策略

1. 先为 Week 1（day01–day07）生成引导文件作为样本
2. 用户审核确认格式和质量
3. 确认后扩展到全部 56 天

## 测试

### Python 测试

- `test_load_guide_returns_none_when_missing` — 无文件时返回 None
- `test_load_guide_merges_into_day` — 有文件时合并到 day 数据
- `test_guide_keys_match_progress` — guide 中的 task key 与 progress.yaml 一致

### 前端测试

- `CurrentFocusPanel` 有 guide 时渲染步骤列表
- `CurrentFocusPanel` 无 guide 时正常降级
- 依赖未完成时显示提示

## 不在范围内

- 引导内容的在线编辑（只读展示）
- 跨天依赖关系
- 自动从引导步骤生成 checklist
