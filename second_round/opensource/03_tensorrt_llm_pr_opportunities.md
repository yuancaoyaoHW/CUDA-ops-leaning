# TensorRT-LLM 开源贡献机会

## 仓库信息
- **Repo**: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **Stars**: 10k+
- **语言**: Python / C++ / CUDA
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/TensorRT-LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/TensorRT-LLM/discussions)

## 候选人优势
- ✅ 有 speculative decoding 经验（TRT-LLM 也支持 EAGLE-3）
- ✅ 熟悉 LLM serving 架构
- ⚠️ 缺少 CUDA 深度（TRT-LLM 核心是 CUDA kernel）
- ⚠️ 缺少 TensorRT 经验

## 候选人短板
- ❌ 无 TensorRT plugin 开发经验
- ❌ 无 CUDA kernel 编写经验（正在学习中）
- ❌ TRT-LLM 代码库复杂度高，入门门槛高

## 相关领域（Relevant Areas）

### 1. Speculative Decoding 方向
TRT-LLM 支持 EAGLE-3（[Issue #8615](https://github.com/NVIDIA/TensorRT-LLM/issues/8615) 提到 EAGLE-3 相关问题），候选人可利用经验参与。

**贡献机会：**
- **EAGLE-3 bug fix**: 修复 speculative decoding 相关 bug
- **Spec decode 文档**: 改进 speculative decoding 使用文档
- **Benchmark 数据**: 补充不同模型的 spec decode benchmark

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Bug fix | Spec decode 相关 issue | hard | 1-2 周 | high |
| Docs PR | Spec decode 配置文档 | easy | 2-3 天 | medium |
| Benchmark PR | EAGLE-3 性能数据 | medium | 3-5 天 | medium |

### 2. 文档与示例方向（最容易切入）
TRT-LLM 文档经常滞后于代码，有大量改进空间。

**贡献机会：**
- **部署教程**: 新模型的部署步骤文档
- **Troubleshooting 文档**: 常见问题解决方案
- **API 文档**: Python API 使用示例补充
- **对比文档**: TRT-LLM vs vLLM 性能对比方法论

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Docs PR | 部署教程改进 | easy | 1-2 天 | low |
| Docs PR | Troubleshooting 补充 | easy | 1-2 天 | low |
| Example PR | 新模型部署示例 | medium | 3-5 天 | medium |

### 3. Benchmark & Testing 方向
**贡献机会：**
- **性能对比**: TRT-LLM vs vLLM vs SGLang 系统化对比
- **回归测试**: 性能回归检测脚本
- **新硬件 benchmark**: 不同 GPU 型号的性能数据

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Benchmark PR | 框架对比 benchmark | medium | 1 周 | high |
| Test PR | 功能测试补充 | medium | 3-5 天 | medium |

### 4. Bug Report & Reproduction
**贡献机会：**
- **Issue 复现**: 帮助复现和定位 bug
- **最小复现脚本**: 为 open issue 提供最小复现
- **环境兼容性**: 测试不同环境下的兼容性

| 类型 | 具体内容 | 难度 | 时间 | 简历价值 |
|------|----------|------|------|----------|
| Bug report | Issue 复现与分析 | medium | 2-3 天 | low |
| Bug fix | 简单 bug 修复 | medium | 3-5 天 | medium |

## Good First Issue 候选（Needs Verification）

TRT-LLM 是 NVIDIA 官方项目，PR 审核严格：
- 文档 typo 和改进
- Python 层面的小 bug fix
- 示例代码补充
- Benchmark 脚本改进

**注意**: TRT-LLM 核心代码（C++/CUDA）的 PR 门槛很高，建议从 Python 层和文档入手。

## 推荐贡献路径

### 短期（1-2 周）
1. **文档 PR** — 改进 speculative decoding 使用文档
2. **Bug 复现** — 帮助复现 EAGLE-3 相关 issue

### 中期（2-4 周）
3. **Benchmark PR** — TRT-LLM vs vLLM 性能对比报告
4. **示例代码** — 新模型部署示例

### 长期（补完 CUDA 后）
5. **Spec decode 优化** — EAGLE-3 在 TRT-LLM 中的性能优化
6. **Plugin 开发** — 自定义 TensorRT plugin

## 风险评估
- **被拒概率**: 中-高（NVIDIA 官方项目，审核严格）
- **审核周期**: 长（2-4 周+）
- **维护成本**: 低（NVIDIA 团队维护）
- **竞争**: 高（关注度高，贡献者多）
- **入门门槛**: 高（代码库复杂，需要 CUDA 基础）

## 面试价值
- **interview value**: ⭐⭐⭐⭐
- 对 NVIDIA 岗位面试直接加分
- 展示对 NVIDIA 生态的了解
- 但需要注意：如果 PR 只是文档级别，面试价值有限

## First Action
1. 安装 TRT-LLM 并跑通一个模型部署
2. 阅读 speculative decoding 相关代码和文档
3. 浏览 [Issues](https://github.com/NVIDIA/TensorRT-LLM/issues) 中 EAGLE/spec decode 相关问题
4. 尝试复现 [Issue #8615](https://github.com/NVIDIA/TensorRT-LLM/issues/8615)（EAGLE-3 infinite loop）
5. 如果能复现并定位原因，提交 bug fix PR
