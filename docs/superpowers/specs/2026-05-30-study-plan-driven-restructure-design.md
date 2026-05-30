# Study-plan-driven 仓库改造设计

日期: 2026-05-30
状态: Draft，待用户审阅

## 1. 改造目标与不动点

### 目标

把 `llm-kernel-lab` 改成 `study-plan/` 真正驱动学习节奏的形态。每天打开仓库的入口是 `python study-plan/run.py day N`，不是 README、不是手动编辑 yaml、也不是 dashboard。

`progress.yaml` 升级为唯一真理源（single source of truth），但它会**自检**：driver 调 `progress.py` 的 verify 引擎，按惯例路径检查文件存在性、JSON 字段、note 内容等，然后把真实状态回写 yaml。

用户的一天（理想状态）：

```
$ python study-plan/run.py day 5
  ── Day 5: row_softmax Nsight Compute ──
  operator: row_softmax  phase: profile  jd_tags: [kernel, perf]
  artifacts: ref ✓  impl ✓  tests ✓  bench ✓  profile ✗  note ✗
  next: run.py day 5 profile

$ python study-plan/run.py day 5 profile
  → bash scripts/run_ncu.sh row_softmax ...

$ python study-plan/run.py day 5 done
  strict verify: profile ✓  note ✗ (notes/row_softmax.md not found)
  ⚠ 5/6 项达标，note 未达 strict；已写 status=profile_stage
  填 daily_check / weaknesses / next_fix 后保存。
```

`done` 不阻塞下一日（软提示模式），但把 `daily_check`、`status`、`weaknesses` 等字段刷新好。

### 不动点（明确不做）

- 不重组 `kernels/` `tests/` `benchmarks/` `notes/` 物理目录（避免 import path 重写）。
- 不回填 `study-plan/week3..8/` 的 56 个 day-level markdown（独立后续 PR）。
- 不引入新的进度系统（只升级现有 `progress.yaml` + `progress.py` + `dashboard.*`）。
- 不动 `cutlass/`、`kernels/cuda_extension/`、`kernels/cuda_ext/` 等 stub 目录（W2 才用）。
- 不写 git hook、不写 CI workflow、不重构 dashboard UI。

### 决策约束（来自 brainstorm）

| 决策点 | 选择 | 含义 |
|---|---|---|
| 改造形态 | D | yaml 自检（A）+ daily driver CLI（C）组合 |
| 验证严格度 | C | 日常文件存在即真；周检/done 跑 strict 内容门槛 |
| Driver 形态 | C | 真·driver，子命令链封装当天命令 |
| 顺序约束 | D | 软提示，不强卡顺序；done 时严格汇总 |
| 命令/路径声明 | C | 约定（zero-config）+ 偏离时 override |
| 门槛位置 | C | progress.py 默认 + operator yaml 可覆盖 |
| Day → operator 映射 | A | 复用 progress.yaml dayN 块，加 operator/phase 字段 |
| 旧算子归属 | C | day00 + Day 1 done 把 4 个 warmup 算子 strict 拉到 6/6 |
| 范围 | B | 核心 + 周检 hooks（STAR / algo / cpp drill），不回填 day md |

## 2. progress.yaml schema 升级

新增四类字段；老字段全部保留兼容。

### meta 块新增 verify_defaults

```yaml
meta:
  verify_defaults:
    note_min_lines: 30
    note_required_sections: ["bottleneck", "next_experiment"]
    bench_required_keys_default: ["gbps"]      # memory-bound 默认
    bench_required_keys_gemm: ["tflops"]       # GEMM 类
    profile_extensions: [".ncu-rep", ".nsys-rep"]
    profile_min_size_bytes: 1024
    pytest_timeout_seconds: 60
```

### operators 块扩展

新增 `kind` 字段（必填，影响默认门槛），`paths` `commands` `thresholds` 全部可选：

```yaml
operators:
  row_softmax:
    kind: reduction              # reduction | gemm | attention | quant | elementwise
    status: not_started
    artifacts:                   # verify 自动写
      reference: false
      implementation: false
      tests: false
      benchmark: false
      profile: false
      note: false
    paths:                       # optional override；缺省走约定
      impl: kernels/triton/row_softmax/row_softmax.py
      tests: tests/test_row_softmax.py
      bench: benchmarks/bench_row_softmax.py
      bench_json: reports/json/row_softmax_bench.json
      profile: reports/ncu/row_softmax.ncu-rep
      note: notes/row_softmax.md
    commands:                    # optional override
      test: pytest tests/test_row_softmax.py -v
    thresholds:                  # optional override，覆盖 verify_defaults
      note_min_lines: 40
    notes: ''
```

### 路径约定（zero-config 默认）

不写 `paths` 时 driver 推断：

| artifact | 默认路径 |
|---|---|
| impl | `kernels/triton/<op>/<op>.py` |
| tests | `tests/test_<op>.py` |
| bench | `benchmarks/bench_<op>.py` |
| bench_json | `reports/json/<op>_bench.json` |
| profile | `reports/ncu/<op>*.ncu-rep` 或 `reports/nsys/bench_<op>*.nsys-rep`（任一命中即可） |
| note | `notes/<op>.md` |

### 命令约定（按 kind 推断）

不写 `commands` 时按 kind 推断默认命令；下表以 reduction 为例：

| phase | 默认命令 |
|---|---|
| reference | 打开 `paths.impl`（`$EDITOR` 或打印路径）+ 提示「写 PyTorch reference 函数」 |
| impl | 打开 `paths.impl` |
| tests | `pytest <paths.tests> -v` |
| bench | `python <paths.bench>` |
| profile | reduction/elementwise/quant → `bash scripts/run_ncu.sh <op>`；attention/serving toy → `bash scripts/run_nsys.sh <op>` |
| note | 打开 `paths.note`；不存在则用模板创建（含 result / bottleneck / next_experiment 段） |

### dayN 块扩展

新增 `operator` `phase`（Q7 A：显式声明，不推断）：

```yaml
days:
  day05:
    title: row_softmax Nsight Compute
    operator: row_softmax
    phase: profile               # reference | implementation | tests | benchmark | profile | note | review
    date: ''
    status: not_started
    daily_check: 0
    jd_tags: [kernel, perf]
    tasks:                       # 用户可勾选的任务，driver 不解析含义
      run_ncu_full: false
      capture_metrics: false
      write_bottleneck_note: false
    artifacts:                   # verify 自动写
      profile: false
      note: false
    verification: ''
    weaknesses: ''
    next_fix: ''
    notes: ''
```

### day00 块（warmup 算子归属）

```yaml
days:
  day00:
    title: Warmup 算子收尾（Day 1 gate）
    operators: [vector_add, axpy, row_sum, row_max]   # 多算子，特殊 day
    phase: review
    date: ''
    status: not_started
    jd_tags: [kernel, perf]
    tasks:
      finish_axpy_note: false
      finish_row_max_note: false
    artifacts: {}
    verification: ''
```

### 周检 day 额外字段

Day 7/14/21/28/35/42/49/56：

```yaml
days:
  day07:
    weekly_check_score: 0          # 0-21
    star_filled: false             # notes/star-weekly.md Week N 是否已填
    algo_drill_done: false         # notes/algorithm-drill.md Week N 算法题
    cpp_drill_done: false          # 同 C++/系统基础题
```

阶段检日（Day 14/28/42/56）保留现有 `stage_check_score: 0`（0-100）。

### Migration

一次性脚本 `study-plan/_migration/migrate_yaml.py`：

1. 把现有 56 天 day 块加上 `operator/phase` 字段（按 `inference-acceleration-plan.md` 表手动对齐）。
2. operators 块加 `kind` 字段（vector_add/axpy/row_sum=reduction，row_max=reduction，row_softmax=reduction，rmsnorm=reduction，flash_attention_toy=attention，int4_dequant=quant；后续算子按需补）。
3. warmup 4 个算子加 day00 块。
4. 写一份 `progress.yaml.bak`。

跑完就归档；不进主路径。

## 3. progress.py 升级为 verify 引擎

把 `progress.py` 从「读 yaml + 渲染」升级为「读 yaml + 验证 + 回写」。新增 3 个子命令，老的全保留。

### 新子命令

```
progress.py verify [--day N | --operator OP | --all] [--strict] [--write] [--skip-tests]
                                                # 不指定 target 时默认 --all
progress.py status [--day N]                     # 单日详细 artifact 状态
progress.py drill                                # 周检 drill 状态总览（star/algo/cpp）
```

### verify 引擎两段式逻辑

```python
def verify(target, *, strict, write, skip_tests=False):
    # target: day N | operator name | "all"
    for op in resolve_operators(target):
        paths = resolve_paths(op)              # 约定 + override
        thresholds = resolve_thresholds(op)    # verify_defaults + kind 默认 + override
        result = {}
        result["reference"]      = check_reference_exists(op, paths)
        result["implementation"] = check_file_exists(paths.impl)
        result["tests"]          = check_tests(paths.tests, strict, skip_tests)
        result["benchmark"]      = check_benchmark(paths.bench_json, thresholds, strict)
        result["profile"]        = check_profile(paths.profile, thresholds, strict)
        result["note"]           = check_note(paths.note, thresholds, strict)
        if write:
            write_back_yaml(op, result, derive_status(result))
    return aggregate(results)
```

### non-strict（日常）vs strict（周检 / done）

| artifact | non-strict | strict |
|---|---|---|
| reference | `paths.impl` 文件存在 | `paths.impl` grep 到 `pytorch_reference\|torch\.` 引用 *或* operator yaml 显式声明 `reference_inline: true` |
| implementation | 文件存在 | 文件存在且非空 |
| tests | `paths.tests` 文件存在 | 跑 `commands.test`（默认 `pytest tests/test_<op>.py -q`），退出码 0；`--skip-tests` 时跳过并标 fail |
| benchmark | `paths.bench_json` 存在 | JSON 存在 + 必填 keys 全部非空（kind 决定 keys；threshold 可覆盖） |
| profile | profile 路径 glob 命中 | glob 命中 `.ncu-rep` 或 `.nsys-rep` 且文件 size > `profile_min_size_bytes` |
| note | `paths.note` 存在 | 行数 ≥ `note_min_lines` 且 markdown 包含 `note_required_sections` 中所有段 |

### status 派生

复用现有 enum（complete / profile_stage / benchmark_stage / not_started 等）：

```
6/6                                  → complete
5/6 含 profile + note 任一缺         → profile_stage 或 note_stage
4/6 含 benchmark                     → benchmark_stage
3/6 含 tests                         → tests_stage
2/6 含 implementation                → impl_stage
≤1/6                                 → not_started
```

### 回写规则

- `--write` 才回写；否则只打印 diff。
- 回写时只动 `operators.<op>.artifacts.*`、`operators.<op>.status`、`days.dayN.artifacts.*`、`days.dayN.status`、`days.dayN.star_filled` / `algo_drill_done` / `cpp_drill_done`（周检 hook）。
- **不碰** `verification` `weaknesses` `next_fix` `notes` `daily_check` `weekly_check_score` `stage_check_score` 这些用户笔记字段。

### analyze 子命令

现有逻辑保留，但底层换成 verify 输出：之前直接读 yaml 静态字段，现在「先 verify(strict=False, write=False) 再统计」。JD 标签覆盖度、算子成熟度、GPU 库覆盖度都变成证据驱动。

### 幂等 + 安全

- 写 yaml 用 `ruamel.yaml`（保留注释和顺序）。如未装则 fallback `PyYAML` + warn 一次。
- 写前生成 `progress.yaml.bak` 备份；连续两次写跳过备份（避免覆盖好备份）。
- verify 不联网、不跑 GPU profile；profile 检查只看文件存在性 + 大小。
- pytest 调用加 60s timeout（来自 `verify_defaults.pytest_timeout_seconds`），超时算 fail。

## 4. Daily driver `study-plan/run.py`

入口：`python study-plan/run.py`，对外子命令：

```
run.py day N                       # 默认动作：show（信息屏）
run.py day N show                  # 显式信息屏
run.py day N <phase>               # phase ∈ {reference, impl, tests, bench, profile, note}
                                   # alias: impl ↔ implementation, bench ↔ benchmark
                                   # yaml dayN.phase 存 full name（implementation / benchmark）
run.py day N done                  # 跑 strict verify，刷新 status，提示填 daily_check
run.py week N                      # 周检入口：列本周 7 天状态 + STAR/algo/cpp drill 状态
run.py week N check                # 跑全周 strict verify + STAR/drill 解析 + 回写
run.py today                       # = run.py day <auto>，按 progress.yaml 已完成天数推断
run.py next                        # 列出下一个 not_started 的天
```

### day N show 输出格式（统一模板）

```
── Day 5 · row_softmax · profile phase ──
date: 2026-06-04   status: benchmark_stage   daily_check: 0/3   jd_tags: kernel perf

operator: row_softmax (kind=reduction)
artifacts (non-strict):
  ✓ reference         kernels/triton/row_softmax/row_softmax.py
  ✓ implementation    kernels/triton/row_softmax/row_softmax.py
  ✓ tests             tests/test_row_softmax.py
  ✓ benchmark         reports/json/row_softmax_bench.json
  ✗ profile           reports/ncu/row_softmax*.ncu-rep   (missing)
  ✗ note              notes/row_softmax.md               (missing)

today's tasks:
  ☐ run_ncu_full           (yaml: tasks.run_ncu_full)
  ☐ capture_metrics
  ☐ write_bottleneck_note

suggested next:
  → run.py day 5 profile
```

### day N <phase> 行为

- 从 operator yaml 取 `commands.<phase>`；无则按 kind 拼默认命令（见 §2 命令约定表）。
- 退出码透传；不强制顺序。
- 跑完后自动跑一次 non-strict verify，回写 day artifacts；不写 daily_check / status。

### day N done

1. 跑 `progress.py verify --day N --strict --write`。
2. 打印未达 strict 的项 + 修复建议（不 block）。
3. 提示：`填 daily_check (0-3) / weaknesses / next_fix 后保存。`（指向 dashboard 或 yaml）。
4. 不自动 commit。

### week N / week N check

`week N`（信息屏）：列本周 7 天 status + STAR/drill chip。

`week N check`（周检日触发）：

1. 全周 strict verify（每天 6 项 artifact）。
2. 检查 `notes/star-weekly.md` 是否有 `## Week N` 段非空 → 写回 `day{N*7}.star_filled`。
3. 检查 `notes/algorithm-drill.md` 是否有 `## Week N` 段含 `### Algo` + `### Cpp` 子段 → 写回 `algo_drill_done` / `cpp_drill_done`。
4. 汇总打印未达项 + 修复建议；不 block。
5. 阶段检日（Day 14/28/42/56）末尾追加：`阶段检: 请填 day{N}.stage_check_score (0-100)`。

### Day 1 gate（warmup 算子）

`run.py day 1 done`（或 `day 0 done`，看 migration 把 gate 挂到哪）触发 strict verify 时，把 vector_add / axpy / row_sum / row_max 一并 strict。任一不过则 warn：

```
⚠ Day 1 gate: 4 个 warmup 算子未到 6/6
  axpy:    note 行数不足 (12 < 30)
  row_max: note 缺 bottleneck/next_experiment 段
建议在进入 row_softmax 主线前补完。
```

不强制阻塞 Day 2（与 Q4 D 软提示一致），只是反复提示。

### 实现规模

单文件 ~300 行，`argparse` + 调 `progress.py` 公开函数。不引入新模块。

## 5. STAR / algorithm-drill 集成 + 周检 hooks

### notes/star-weekly.md 解析约定

文件已存在；driver 用以下规则识别：

| 检查 | 规则 |
|---|---|
| 段存在 | grep `^## Week N\b` 命中 |
| 段非空 | 该段下面到下一个 `^## Week` 之间，去掉模板占位文本「(待填)」后 ≥ 5 个非空行 |
| strict（Day 56 才检） | 段内必须出现 `Situation` `Task` `Action` `Result` `Badcase` 五个子标题 |

回写 `progress.yaml.days.day{N*7}.star_filled = true/false`。

### notes/algorithm-drill.md（新建）

模板：

```markdown
# 算法 / C++ Drill 周记

每周周检（Day 7/14/21/28/35/42/49/56）做 1 道算法 + 1 道 C++/系统基础。
60-90 分钟内完成；不抢主线。drill 表见 study-plan/inference-acceleration-plan.md。

模板：

## Week N (YYYY-MM-DD)

### Algo: <题名>
- 题目链接 / 出处:
- 思路（一句话）:
- 复杂度: O(...)
- 关键代码片段:
  ```cpp
  // ...
  ```
- 卡点 / 复盘:

### Cpp: <主题>
- 主题（如 RAII / smart pointer）:
- 学到的 1 个反直觉点:
- 一段最小代码示例:
  ```cpp
  // ...
  ```
- 自测题（口述）:

---

## Week 1 (待填)
## Week 2 (待填)
## Week 3 (待填)
## Week 4 (待填)
## Week 5 (待填)
## Week 6 (待填)
## Week 7 (待填)
## Week 8 (待填)
```

driver 检查：

| 字段 | 规则 |
|---|---|
| `algo_drill_done` | `## Week N` 段下有 `### Algo:` 子标题 + 该子段 ≥ 3 个非空行（去掉 "待填"） |
| `cpp_drill_done` | 同上但 `### Cpp:` 子标题 |

### `week N check` 完整流程

1. 全周 strict verify（每天 6 项 artifact）。
2. 解析 `star-weekly.md` Week N → 回写 `day{N*7}.star_filled`。
3. 解析 `algorithm-drill.md` Week N → 回写 `day{N*7}.algo_drill_done` / `cpp_drill_done`。
4. 汇总打印：

```
── Week 1 Check ──
strict verify: 5/7 days at strict pass
  day02 row_softmax  ✗ note 行数不足 (12 < 30)
  day05 row_softmax  ✗ profile 缺失
STAR Week 1:    ✓ filled (62 lines)
Algo drill:     ✗ Week 1 段为空（"待填"）
Cpp drill:      ✗ Week 1 段为空
未达项:
  - 写 notes/algorithm-drill.md Week 1 ### Algo / ### Cpp
  - 补 day05 ncu profile
下一步: run.py week 1 check 重跑确认；然后填 day07.weekly_check_score
```

5. 不 block，不自动 commit；只汇报。

## 6. Dashboard 适配 + 测试覆盖

### 读取侧（`dashboard.py` / `dashboard.html` / `server.py`）

- `dashboard.py --build` 调用 `progress.verify(write=False)` 拿真实 artifact 状态而非 yaml 静态字段；同一份 yaml，dashboard 显示的勾比 yaml 字段更"诚实"。
- 新增列：`operator`、`phase`、`star_filled`、`algo_drill_done`、`cpp_drill_done`（周检日才显示后三列）。
- 算子成熟度卡片新增 `kind` 标签 + 区分 non-strict / strict 两种状态的进度条。

### 写入侧（`server.py` 编辑接口）

- 编辑 yaml 时**只允许写「用户笔记字段」**：`daily_check` `weaknesses` `next_fix` `verification` `notes` `weekly_check_score` `stage_check_score`。
- artifact 字段（`artifacts.*`、`status`、`star_filled` / `algo_drill_done` / `cpp_drill_done`）一律拒绝编辑，dashboard 只读；想改就让用户去补真文件然后 `run.py day N done`。
- 这是为了让 yaml 的 artifacts 永远是 verify 真值，避免老路（手填 true 自欺）。

### UI 提示

- 每天卡片顶部加一行 `→ run.py day N show / done` 命令提示。
- 周检日卡片高亮（黄/绿色），把 STAR / algo / cpp drill 三栏单独 chip。

### 测试覆盖

| 测试模块 | 覆盖点 |
|---|---|
| `tests/test_progress_verify.py`（新建） | 路径约定推断；strict vs non-strict 行为；6 项 artifact 各自的 pass/fail case；status 派生表；回写 yaml 不破坏注释；fallback PyYAML 路径；pytest 超时；`--skip-tests` 行为 |
| `tests/test_run_driver.py`（新建） | `day N show` 输出包含必要字段；`day N <phase>` 调用对的 command；`day N done` 不 block 但 warn；`week N check` 三段汇总顺序；STAR/drill 解析正则 |
| `tests/test_study_plan_dashboard.py`（已存在，扩展） | 新增字段渲染；server.py 拒绝写 artifact 字段；周检日高亮 |
| 数据 fixtures | `tests/fixtures/study-plan/` 放迷你 yaml + 假 reports/json/*.json + 假 ncu-rep（空文件 + 大文件）+ 假 notes md，避免污染真 progress.yaml |

CI / 命令：

- `bash scripts/04_verify_all.sh` 末尾追加 `pytest tests/test_progress_verify.py tests/test_run_driver.py tests/test_study_plan_dashboard.py -q`。
- 不引入新 CI 服务（本地仓库无远程 CI）。

## 7. 数据流 / 错误处理 / 落地切换

### 端到端数据流

```
用户敲 run.py day 5 done
        │
        ▼
run.py: parse args → import progress
        │
        ▼
progress.verify(target=day5, strict=True, write=True)
        │
        ├─ load progress.yaml (ruamel.yaml 优先)
        ├─ resolve_paths(op=row_softmax)        # 约定 + override
        ├─ resolve_thresholds(op)               # verify_defaults + kind 默认 + override
        ├─ run 6 个 check_* 函数
        │     ├─ check_tests: subprocess pytest（strict 才跑，超时 60s）
        │     ├─ check_benchmark: 读 JSON，校验 keys
        │     ├─ check_profile: glob + size > 1KB
        │     └─ check_note: 行数 + 必填段 grep
        ├─ derive_status(results) → benchmark_stage
        ├─ write_back: operators.row_softmax.{artifacts, status}
        │              days.day05.{artifacts, status}
        │              （备份 progress.yaml.bak）
        └─ return aggregate
        │
        ▼
run.py: 打印 strict 汇总；列未达项 + 修复建议；
        提示「填 daily_check / weaknesses / next_fix」
        退出码 0（不 block）
```

### 错误处理矩阵

| 失败点 | 处理 | 退出码 |
|---|---|---|
| yaml 不存在 / 损坏 | 打印路径 + 提示恢复 `progress.yaml.bak` | 2 |
| 算子 yaml 块缺失（day 引用了未声明的 op） | 打印缺失算子名 + 建议在 `operators:` 加块 | 2 |
| pytest 命令本身不存在（环境问题） | strict tests=False，verify 继续；汇总里标 `tests: ✗ (pytest unavailable)` | 0 |
| pytest 超时（>60s） | 视为 fail + warn `consider --skip-tests`；不挂死 | 0 |
| benchmark JSON 字段缺 | strict bench=False，列出缺失 keys | 0 |
| 备份写失败（磁盘满 / 权限） | 中止写回 + 保留原 yaml + 报错 | 3 |
| ruamel/PyYAML 都没装 | 报错让用户装 | 4 |
| `run.py day 99` 越界 | 报错列出有效范围 1-56 + day00 | 2 |
| `progress.py verify --operator NotExist` | 报错列出已知 operator 名 | 2 |
| GPU/profile 命令外部失败 | run.py 透传退出码；不影响下一次 verify | 透传 |

### Rollout / 切换策略

改造分两阶段提交：

**阶段 A — schema + verify**：

1. 升级 `progress.yaml` schema（meta / operators / dayN / day00 / 周检字段）。
2. 写 migration 脚本 `study-plan/_migration/migrate_yaml.py`，跑一次后归档。
3. 写 `progress.py verify` 引擎 + `status` / `drill` 子命令；`analyze` 切换底层。
4. 扩 dashboard 读侧 + server.py 写入限制。
5. 补 `tests/test_progress_verify.py` + 扩 `tests/test_study_plan_dashboard.py`。

阶段 A 结束时：现有 dashboard 在新 schema 下正常工作；用户仍按老方式手动跑命令，但 `progress.py verify` 已经能给出真值。

**阶段 B — driver + 周检 hooks**：

6. 写 `study-plan/run.py`（`day N show/<phase>/done`、`week N show/check`、`today`、`next`）。
7. 新建 `notes/algorithm-drill.md` 模板。
8. 在 `progress.py` 加 STAR / drill 解析。
9. dashboard 加周检日高亮 + drill chip。
10. 补 `tests/test_run_driver.py`。
11. 更新 `04_verify_all.sh` 跑新测试。

### 文档同步（必做小项）

- 更新根 `README.md`：把入口指到 `python study-plan/run.py today`，移除老的"8 周泛 AI infra"叙事。
- 更新 `study-plan/README.md`：新增 driver / verify 一节，提到 `progress.yaml` 是真理源、artifact 字段不再手填。
- 更新 `AGENTS.md`：在 "Required Verification" 里追加 "After changing operator artifacts, run `python study-plan/progress.py verify --operator <op>`"。

### 不做 / 推迟

- 不回填 56 个 day-level markdown（独立后续 PR）。
- 不写 git hook（不在 commit 时触发 verify）。
- 不写 CI workflow。
- 不动 `cutlass/` `kernels/cuda_extension/` 的 stub。
- dashboard 不重构 UI，只加字段 / 限制写入。

## 8. 验收标准

阶段 A 完成时：

- [ ] `python study-plan/progress.py verify --operator row_sum --strict` 输出 6/6（已有完整证据）。
- [ ] `python study-plan/progress.py verify --operator axpy --strict` 输出 5/6 或 4/6 并明确指出 note 缺哪段（partial 状态）。
- [ ] `python study-plan/progress.py verify --operator row_softmax` 输出 0/6（尚未开始）。
- [ ] `python study-plan/progress.py verify --all --write` 跑完后 `git diff progress.yaml` 只动 verify 写回字段（`artifacts.*`、`status`、`star_filled` / `algo_drill_done` / `cpp_drill_done`），不动用户笔记字段。
- [ ] `python study-plan/dashboard.py --build` 渲染新字段不报错。
- [ ] `tests/test_progress_verify.py` 全过。

阶段 B 完成时：

- [ ] `python study-plan/run.py today` 正确推断当前应做的 day。
- [ ] `python study-plan/run.py day 5 show` 输出符合 §4 的模板。
- [ ] `python study-plan/run.py day 1 done` 报 warmup 算子未达 strict，列出具体不通过项。
- [ ] `python study-plan/run.py week 1 check` 三段汇总（strict verify + STAR + drill）按顺序输出。
- [ ] `bash scripts/04_verify_all.sh` 末尾 pytest 全过。
- [ ] 根 README 入口指向 driver。
