# Triton 开源贡献机会

## 仓库信息
- **Repo**: [triton-lang/triton](https://github.com/triton-lang/triton)
- **Stars**: 15k+
- **语言**: Python / C++ / MLIR
- **文档**: [Triton Documentation](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- **教程**: [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- **扩展**: [triton-lang/triton-ext](https://github.com/triton-lang/triton-ext)

## 候选人优势
- ✅ 有 Python 编程基础（Triton 是 Python DSL）
- ✅ 理解 LLM 推理算子（softmax, attention, GEMM）
- ✅ 正在学习 CUDA（Triton 是 CUDA 的高层抽象）
- ⚠️ 无 Triton 编程经验（需要学习）
- ⚠️ 无编译器/MLIR 经验

## 候选人短板
- ❌ 无 Triton kernel 编写经验
- ❌ 无 MLIR/编译器背景（Triton 核心是编译器）
- ❌ Triton 核心开发门槛极高（需要编译器知识）

## 相关领域（Relevant Areas）

### 1. Triton Tutorials & Documentation（最容易切入）
Triton 教程是学习和贡献的最佳切入点。

**贡献机会：**
- **新教程**: 编写 LLM 推理相关的 Triton kernel 教程
- **教程改进**: 改进现有教程的解释和注释
- **性能对比**: Triton vs CUDA 的性能对比文档
- **最佳实践**: Triton kernel 优化最佳实践文档

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Docs PR | 新增 RMSNorm Triton 教程 | medium | 3-5 天 | medium |
| Docs PR | 新增 RoPE Triton 教程 | medium | 3-5 天 | medium |
| Docs PR | Triton vs CUDA 性能对比 | medium | 1 周 | medium |
| Docs PR | 教程注释改进 | easy | 1-2 天 | low |

### 2. triton-ext 扩展库（门槛较低）
[triton-lang/triton-ext](https://github.com/triton-lang/triton-ext) 是 out-of-tree 扩展集合，门槛比主仓库低。

**贡献机会：**
- **新 kernel 实现**: 贡献 LLM 推理相关的 Triton kernel
- **性能优化**: 优化现有 kernel 的性能
- **测试补充**: 补充 kernel 正确性和性能测试

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Feature PR | LLM 推理 kernel（softmax/layernorm） | medium | 1 周 | high |
| Performance PR | 现有 kernel 优化 | hard | 1-2 周 | high |
| Test PR | Kernel 正确性测试 | easy | 2-3 天 | low |

### 3. Bug Report & Reproduction
**贡献机会：**
- **Bug 复现**: 帮助复现编译器 bug
- **性能回归**: 报告 Triton 版本间的性能回归
- **兼容性测试**: 不同 GPU 架构的兼容性测试

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Bug report | 性能回归报告 | medium | 2-3 天 | low |
| Test PR | 兼容性测试 | medium | 3-5 天 | low |

### 4. Benchmark & Performance Analysis
**贡献机会：**
- **Kernel benchmark**: 不同 Triton kernel 的性能 benchmark
- **编译时间分析**: Triton JIT 编译时间优化建议
- **Autotuning 改进**: 更好的 autotuning 配置

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Benchmark PR | LLM kernel benchmark suite | medium | 1 周 | medium |
| Performance PR | Autotuning 配置优化 | medium | 3-5 天 | medium |

## Good First Issue 候选（Needs Verification）

Triton 主仓库的贡献门槛较高（编译器方向），建议：
- 从教程和文档入手
- 在 triton-ext 贡献 kernel
- 报告 bug 和性能问题
- 贡献 benchmark 数据

## 推荐贡献路径

### 短期（1-2 周）— 学习阶段
1. **完成 Triton 官方教程** — 实现 softmax, matmul, attention
2. **写学习笔记** — 可作为教程 PR 的基础

### 中期（2-4 周）— 贡献阶段
3. **triton-ext kernel PR** — 贡献 RMSNorm/RoPE Triton kernel
4. **性能对比文档** — Triton vs CUDA vs PyTorch 性能对比

### 长期（1-2 月）
5. **FlashAttention Triton 实现** — 在 Triton 中实现简化版 FlashAttention
6. **教程 PR** — 贡献完整的 LLM inference kernel 教程

## 风险评估
- **被拒概率**: 中（主仓库高，triton-ext 低）
- **审核周期**: 中等（1-2 周）
- **维护成本**: 低（kernel 相对独立）
- **入门门槛**: 中（Python DSL 降低了门槛，但需要理解 GPU 编程模型）

## 面试价值
- **interview value**: ⭐⭐⭐⭐
- Triton 是越来越多团队的首选 kernel 开发工具
- 展示从 CUDA 到 Triton 的全栈能力
- Meta、NVIDIA、OpenAI 等公司大量使用 Triton
- 但注意：Triton 贡献需要先有 Triton 编程基础

## First Action
1. 完成 [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) 中的 vector_add, softmax, matmul
2. 实现 RMSNorm 和 RoPE 的 Triton 版本
3. 对比 Triton 实现与 PyTorch native 的性能
4. 将学习成果整理为可提交的教程 PR
5. 浏览 [triton-ext](https://github.com/triton-lang/triton-ext) 了解已有扩展
