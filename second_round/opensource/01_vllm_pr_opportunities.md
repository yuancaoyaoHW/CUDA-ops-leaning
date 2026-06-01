# vLLM 开源贡献机会

## 仓库信息
- **Repo**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Stars**: 50k+
- **语言**: Python / CUDA C++
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/vllm-project/vllm/blob/main/CONTRIBUTING.md)
- **Roadmap**: [2026 Q1 Roadmap](https://github.com/vllm-project/vllm/issues/32455)

## 候选人优势
- ✅ 已有 vLLM-Ascend PR #1032 合入经验（EAGLE-3 proposer）
- ✅ 熟悉 vLLM 代码结构（scheduler, attention backend）
- ✅ 熟悉 speculative decoding 流程
- ✅ 有 vLLM-MindSpore PR #1020 提交经验

## 相关领域（Relevant Areas）

### 1. Speculative Decoding 方向（最匹配）
候选人已有 EAGLE-3 实现经验，可直接参与主仓库 speculative decoding 改进。

**贡献机会：**
- **speculators 库集成**: [vllm-project/speculators](https://github.com/vllm-project/speculators) 是 vLLM 新推出的统一 speculative decoding 库，可贡献新 proposer 实现
- **EAGLE 3.1 相关**: vLLM 官方博客 [EAGLE 3.1](https://vllm.ai/blog/2026-05-26-eagle-3-1) 刚发布，可能有后续优化和 bug fix 机会
- **Spec decode benchmark**: 补充不同模型/硬件组合的 benchmark 数据
- **文档改进**: speculative decoding 使用文档、参数调优指南

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Performance PR | 优化 spec decode 的 verification 效率 | hard | 2-3 周 | high |
| Benchmark PR | 补充 EAGLE-3 在不同模型上的 benchmark | medium | 3-5 天 | medium |
| Docs PR | 完善 speculative decoding 配置文档 | easy | 1-2 天 | low |
| Test PR | 补充 spec decode edge case 测试 | easy | 2-3 天 | low |

### 2. Attention Backend 方向
**贡献机会：**
- **FlashInfer backend 优化**: vLLM 正在深度集成 FlashInfer，可参与适配工作
- **Prefix caching 改进**: prefix cache hit rate 相关优化
- **Multi-modal attention**: 多模态模型的 attention 适配

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Performance PR | prefix cache 命中率优化 | hard | 2-3 周 | high |
| Bug fix | attention backend 兼容性问题 | medium | 3-5 天 | medium |
| Test PR | attention 精度验证测试 | easy | 2-3 天 | low |

### 3. Benchmark & Testing 方向
**贡献机会：**
- **benchmark 脚本改进**: 现有 benchmark 脚本的可用性改进
- **CI 测试覆盖**: 补充缺失的单元测试
- **性能回归检测**: 帮助建立性能回归 CI

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Benchmark PR | 新增 workload pattern benchmark | medium | 3-5 天 | medium |
| Test PR | 补充 scheduler 单元测试 | easy | 2-3 天 | low |
| Docs PR | benchmark 使用说明改进 | easy | 1 天 | low |

### 4. Hardware Backend 方向（利用 Ascend 经验）
**贡献机会：**
- **vLLM-Ascend 持续贡献**: 继续在 vLLM-Ascend 仓库贡献
- **跨平台兼容性**: 帮助改进 vLLM 的硬件抽象层
- **AMD ROCm 适配**: 利用非 NVIDIA 硬件经验参与 ROCm 适配

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Feature PR | vLLM-Ascend 新功能适配 | medium | 1-2 周 | high |
| Bug fix | 跨平台兼容性修复 | medium | 3-5 天 | medium |

## Good First Issue 候选（Needs Verification）

以下类型的 issue 通常适合新贡献者：
- 标签 `good first issue` 的 issue
- 文档 typo 和改进
- 测试覆盖补充
- 小型 bug fix（特别是 edge case）
- benchmark 脚本改进

**注意**: vLLM 社区 PR 审核较慢（[社区讨论](https://discuss.vllm.ai/t/why-there-are-so-many-open-pull-requests/2317)），建议：
1. 先在 issue 中表明意图
2. 选择有明确 scope 的小 PR
3. 优先选择有 maintainer 回复的 issue

## 推荐贡献路径

### 短期（1-2 周）
1. **speculators 库文档 PR** — 基于 EAGLE-3 经验补充使用文档
2. **spec decode benchmark 数据** — 补充不同配置下的性能数据

### 中期（2-4 周）
3. **vLLM-Ascend 新功能** — 继续在 Ascend 生态贡献
4. **spec decode 优化 PR** — 基于 profiling 发现的优化点

### 长期（1-2 月）
5. **主仓库 speculative decoding 改进** — 新 proposer 算法或 verification 优化

## 风险评估
- **被拒概率**: 低（已有合入记录，社区认可）
- **审核周期**: 中等偏长（2-4 周）
- **维护成本**: 低（spec decode 方向相对独立）
- **竞争**: 中等（spec decode 方向关注度高）

## 面试价值
- **interview value**: ⭐⭐⭐⭐⭐
- 可以讲述"从 vLLM-Ascend 到 vLLM 主仓库"的贡献升级故事
- EAGLE 3.1 刚发布，时效性强
- 展示对 speculative decoding 的深入理解

## First Action
1. 阅读 [vllm-project/speculators](https://github.com/vllm-project/speculators) 仓库代码
2. 阅读 [EAGLE 3.1 博客](https://vllm.ai/blog/2026-05-26-eagle-3-1) 了解最新进展
3. 在 vLLM Discord 或 GitHub Discussion 中了解当前 spec decode 方向的 open issues
4. 选择一个文档或 benchmark PR 作为切入点
