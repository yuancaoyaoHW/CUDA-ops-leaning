# SGLang 开源贡献机会

## 仓库信息
- **Repo**: [sgl-project/sglang](https://github.com/sgl-project/sglang)
- **Stars**: 20k+
- **语言**: Python / CUDA C++
- **Contributing Guide**: [Contribution Guide](https://sgl-project.github.io/developer_guide/contribution_guide.html)
- **Roadmap**: [2026 Q2 Development Roadmap](https://roadmap.sglang.io/)
- **NVIDIA Collaboration**: [SGLang × NVIDIA Roadmap](https://github.com/sgl-project/sglang/issues/17130)
- **Ascend 贡献指南**: [Ascend Contribution Guide](https://sgl-project.github.io/platforms/ascend/ascend_contribution_guide.html)

## 候选人优势
- ✅ 有 vLLM 经验，SGLang 架构类似但有差异化设计
- ✅ 熟悉 speculative decoding（SGLang 也在积极开发）
- ✅ 有 Ascend NPU 经验（SGLang 有 Ascend 适配方向）
- ✅ 熟悉 LLM serving 核心概念（RadixAttention, continuous batching）

## 相关领域（Relevant Areas）

### 1. Ascend 平台适配（最直接切入点）
SGLang 有专门的 [Ascend 贡献指南](https://sgl-project.github.io/platforms/ascend/ascend_contribution_guide.html)，候选人可直接利用 NPU 经验。

**贡献机会：**
- **Ascend backend 功能补全**: 新模型适配、算子支持
- **Ascend 性能优化**: NPU 上的 serving 性能调优
- **Ascend CI/测试**: 补充 Ascend 平台测试覆盖

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Feature PR | Ascend 新模型适配 | medium | 1-2 周 | high |
| Performance PR | Ascend serving 性能优化 | hard | 2-3 周 | high |
| Test PR | Ascend 平台测试补充 | easy | 2-3 天 | medium |
| Docs PR | Ascend 部署文档改进 | easy | 1-2 天 | low |

### 2. Speculative Decoding 方向
SGLang 也在积极开发 speculative decoding 支持，候选人可迁移 EAGLE-3 经验。

**贡献机会：**
- **EAGLE 系列集成**: 将 EAGLE-3/3.1 适配到 SGLang
- **Spec decode benchmark**: 对比 SGLang vs vLLM 的 spec decode 性能
- **新 proposer 实现**: 在 SGLang 框架下实现新的 draft model 策略

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Feature PR | EAGLE proposer 适配 | hard | 2-4 周 | high |
| Benchmark PR | Spec decode 性能对比 | medium | 3-5 天 | medium |
| Docs PR | Spec decode 使用文档 | easy | 1-2 天 | low |

### 3. Benchmark & Performance 方向
**贡献机会：**
- **Benchmark suite 改进**: SGLang 的 benchmark 工具改进
- **性能回归检测**: 帮助建立性能 CI
- **Workload 模拟**: 新增真实 workload pattern 的 benchmark

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Benchmark PR | 新增 benchmark 场景 | medium | 3-5 天 | medium |
| Performance PR | RadixAttention 缓存优化 | hard | 2-3 周 | high |
| Test PR | 性能回归测试 | medium | 3-5 天 | medium |

### 4. 文档与教程方向
**贡献机会：**
- **部署教程**: 不同硬件/场景的部署指南
- **架构文档**: SGLang 内部架构解析
- **对比文档**: SGLang vs vLLM 特性对比

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Docs PR | 部署最佳实践文档 | easy | 1-2 天 | low |
| Docs PR | 架构设计文档 | medium | 3-5 天 | medium |

### 5. MLX Support（2026 Q1 Roadmap）
[Issue #19137](https://github.com/sgl-project/sglang/issues/19137) — SGLang 正在开发 MLX 支持，这是一个新方向。

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Feature PR | MLX backend 贡献 | hard | 3-4 周 | medium |

## Good First Issue 候选（Needs Verification）

SGLang 社区相对 vLLM 更小，PR 审核可能更快：
- 文档改进和 typo 修复
- 测试覆盖补充
- 小型 bug fix
- Benchmark 脚本改进
- Ascend 平台相关 issue

## 推荐贡献路径

### 短期（1-2 周）
1. **Ascend 平台文档 PR** — 利用 NPU 经验改进 Ascend 部署文档
2. **Benchmark 对比** — SGLang vs vLLM 在相同配置下的性能对比

### 中期（2-4 周）
3. **Ascend 新模型适配** — 在 SGLang-Ascend 上适配新模型
4. **Spec decode 功能** — 将 EAGLE 经验迁移到 SGLang

### 长期（1-2 月）
5. **RadixAttention 优化** — 深入 SGLang 核心调度优化
6. **EAGLE proposer 完整实现** — 在 SGLang 中实现完整的 EAGLE-3

## 风险评估
- **被拒概率**: 低-中（社区活跃，对新贡献者友好）
- **审核周期**: 较快（1-2 周，社区较小）
- **维护成本**: 中（SGLang 迭代快，API 可能变化）
- **竞争**: 中（Ascend 方向竞争较少）

## 面试价值
- **interview value**: ⭐⭐⭐⭐⭐
- 展示对多个 serving framework 的理解（vLLM + SGLang）
- 可对比两个框架的设计哲学差异
- SGLang 是 Together AI / Fireworks AI 等公司关注的框架
- 展示跨框架迁移能力

## First Action
1. 阅读 [SGLang Contribution Guide](https://sgl-project.github.io/developer_guide/contribution_guide.html)
2. 阅读 [Ascend Contribution Guide](https://sgl-project.github.io/platforms/ascend/ascend_contribution_guide.html)
3. 浏览 [2026 Q2 Roadmap](https://roadmap.sglang.io/) 了解当前开发重点
4. Fork 仓库，本地搭建开发环境
5. 选择 Ascend 方向的文档或小 feature 作为第一个 PR
