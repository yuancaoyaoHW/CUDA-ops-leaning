# FlashAttention & FlashInfer 开源贡献机会

## 仓库信息

### FlashAttention
- **Repo**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **Stars**: 15k+
- **语言**: CUDA C++ / Python
- **维护者**: Tri Dao (Princeton/Together AI)

### FlashInfer
- **Repo**: [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **Stars**: 3k+
- **语言**: CUDA C++ / Python (TVM FFI)
- **Issues**: [325+ open issues](https://github.com/flashinfer-ai/flashinfer/issues)
- **PRs**: [200+ open PRs](https://github.com/flashinfer-ai/flashinfer/pulls)
- **Bench**: [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench)

## 候选人优势
- ✅ 理解 attention 机制和 KV cache
- ✅ 有 vLLM attention backend 经验
- ✅ 正在学习 CUDA kernel 编写
- ⚠️ 无 FlashAttention kernel 编写经验
- ⚠️ 无 TVM/FFI 经验（FlashInfer 使用）

## 候选人短板
- ❌ FlashAttention 核心 CUDA kernel 门槛极高（Tri Dao 级别）
- ❌ 无 CUTLASS 经验
- ❌ FlashInfer 使用 TVM FFI，需要额外学习

---

## FlashAttention 贡献机会

### 1. Benchmark & Testing（最容易切入）
**贡献机会：**
- **性能 benchmark**: 不同 head_dim, seq_len, batch_size 的性能数据
- **正确性测试**: 边界条件测试补充
- **新硬件测试**: 不同 GPU 架构的兼容性测试

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Benchmark PR | 系统化性能 benchmark | medium | 3-5 天 | medium |
| Test PR | 边界条件测试 | easy | 2-3 天 | low |
| Bug report | 兼容性问题报告 | easy | 1-2 天 | low |

### 2. Python Binding & Interface
**贡献机会：**
- **API 改进**: Python 接口易用性改进
- **文档**: API 使用文档和示例
- **集成示例**: 与 PyTorch/vLLM 的集成示例

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Docs PR | API 使用文档 | easy | 1-2 天 | low |
| Example PR | 集成示例代码 | medium | 3-5 天 | medium |

### 3. 核心 Kernel（门槛极高）
**注意**: FlashAttention 核心 kernel 由 Tri Dao 主导，外部贡献极难被接受。

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Performance PR | Kernel 优化 | extreme | 1+ 月 | extreme |
| Feature PR | 新 attention variant | extreme | 1+ 月 | extreme |

---

## FlashInfer 贡献机会（更推荐）

FlashInfer 社区更活跃，issue/PR 数量多，贡献机会更多。

### 1. Benchmark & Performance（最容易切入）
[flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) 是专门的 benchmark 仓库。

**贡献机会：**
- **Benchmark 脚本**: 新增 benchmark 场景
- **性能对比**: FlashInfer vs FlashAttention 对比
- **Profiling 报告**: 不同配置下的性能分析

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Benchmark PR | 新增 benchmark 场景 | medium | 3-5 天 | medium |
| Benchmark PR | 性能对比报告 | medium | 1 周 | high |
| Docs PR | Benchmark 使用文档 | easy | 1-2 天 | low |

### 2. Python API & 兼容性
FlashInfer 有 PyTorch 兼容性问题（[Issue #1965](https://github.com/flashinfer-ai/flashinfer/issues/1965)）。

**贡献机会：**
- **兼容性修复**: PyTorch 版本兼容性
- **安装改进**: 安装流程简化
- **API 文档**: 使用文档改进

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Bug fix | PyTorch 兼容性修复 | medium | 3-5 天 | medium |
| Docs PR | 安装和使用文档 | easy | 1-2 天 | low |
| Feature PR | 新 API wrapper | medium | 3-5 天 | medium |

### 3. Serving Integration
FlashInfer 被 vLLM 和 SGLang 使用，可以从集成角度贡献。

**贡献机会：**
- **vLLM 集成优化**: 改进 FlashInfer 在 vLLM 中的使用
- **SGLang 集成**: FlashInfer 在 SGLang 中的适配
- **新 attention pattern**: 支持新的 attention 变体

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Feature PR | 新 attention pattern 支持 | hard | 2-3 周 | high |
| Performance PR | Serving 场景优化 | hard | 2-3 周 | high |
| Test PR | 集成测试 | medium | 3-5 天 | medium |

### 4. 文档与教程
**贡献机会：**
- **架构文档**: FlashInfer 内部架构解析
- **使用教程**: 不同场景的使用指南
- **迁移指南**: 从 FlashAttention 迁移到 FlashInfer

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Docs PR | 使用教程 | easy | 1-2 天 | low |
| Docs PR | 架构文档 | medium | 3-5 天 | medium |

## 推荐贡献路径

### 短期（1-2 周）
1. **FlashInfer benchmark PR** — 在 flashinfer-bench 贡献 benchmark 场景
2. **FlashInfer 文档 PR** — 改进安装和使用文档

### 中期（2-4 周）
3. **FlashInfer 兼容性修复** — 解决 PyTorch 版本兼容问题
4. **性能对比报告** — FlashInfer vs FlashAttention 系统化对比

### 长期（补完 CUDA 后）
5. **FlashInfer kernel 贡献** — 新 attention pattern 实现
6. **FlashAttention benchmark** — 系统化性能测试

## 风险评估

### FlashAttention
- **被拒概率**: 高（核心 kernel PR 几乎不接受外部贡献）
- **审核周期**: 长
- **入门门槛**: 极高

### FlashInfer
- **被拒概率**: 中（社区更开放）
- **审核周期**: 中等（1-2 周）
- **入门门槛**: 中-高（需要理解 TVM FFI）
- **维护成本**: 中

## 面试价值
- **FlashAttention interview value**: ⭐⭐⭐⭐⭐（但贡献难度极高）
- **FlashInfer interview value**: ⭐⭐⭐⭐
- 展示对 attention kernel 的深入理解
- FlashInfer 是 vLLM/SGLang 的核心依赖
- 可以讲述"从使用者到贡献者"的故事

## First Action
1. 安装 FlashInfer 并跑通基础 benchmark
2. 阅读 [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) 了解现有 benchmark
3. 尝试复现 [Issue #1965](https://github.com/flashinfer-ai/flashinfer/issues/1965)（PyTorch 兼容性）
4. 在 flashinfer-bench 贡献新的 benchmark 场景
5. 学习 FlashAttention 论文，理解 tiling 策略
