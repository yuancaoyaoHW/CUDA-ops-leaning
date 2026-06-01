# 国内 AI Infra 技术画像

---

## 1. 华为昇腾

### 基本信息
- **总部**: 深圳/杭州/北京
- **规模**: 大型（昇腾事业部数千人）
- **特点**: 国产 AI 芯片 + 全栈软件生态

### Infra Product
- **硬件**: Atlas 300/800/900 系列（Ascend 910B/910C）
- **框架**: MindSpore, CANN (Compute Architecture for Neural Networks)
- **推理**: MindIE (MindSpore Inference Engine)
- **适配**: vLLM-Ascend, SGLang-Ascend
- **集群**: Atlas 3000 训练集群

### Likely Stack
- **推理框架**: MindIE + vLLM-Ascend + 自研
- **算子库**: CANN Ascend C 算子
- **编程语言**: Python, C++, Ascend C
- **通信**: HCCL (Huawei Collective Communication Library)
- **调度**: MindX DL (K8s-based)
- **Profiling**: msprof (MindSpore Profiler)

### Target Roles
1. **推理引擎开发工程师** — MindIE/vLLM-Ascend 开发
2. **大模型推理优化工程师** — 推理性能优化
3. **算子开发工程师** — Ascend C 算子开发
4. **AI 框架工程师** — MindSpore 框架开发

### Interview Focus
- vLLM/推理框架内部机制
- Speculative decoding 原理和实现
- KV cache 管理
- NPU 编程模型（Ascend C）
- 性能优化方法论
- 算法题（LeetCode Medium）

### Resume Angle
- **核心优势**: vLLM-Ascend PR #1032 合入（EAGLE-3 proposer）
- 突出 NPU 适配经验
- 突出 55% throughput improvement 数据
- 突出 speculative decoding 深度理解

### Project Evidence Needed
- ✅ 已有：vLLM-Ascend PR 合入
- ✅ 已有：EAGLE-3 实现经验
- ✅ 已有：Ascend NPU 性能优化数据
- 加分：更多 vLLM-Ascend 贡献

### Risk
- **面试难度**: 中（最匹配的岗位）
- **面试通过率**: 85%（极高）
- **薪资**: 中等（国内水平）
- **成长空间**: 中-高（NPU 生态发展中）

### Application Strategy
- **P0 — 立即投递**
- 通过 vLLM-Ascend PR reviewer 内推
- 中文简历突出昇腾经验
- 面试重点准备 EAGLE-3 STAR 故事

---

## 2. 阿里云 PAI

### 基本信息
- **总部**: 杭州/北京
- **规模**: 大型（PAI 团队数百人）
- **特点**: 国内最大云 AI 平台之一

### Infra Product
- **PAI-EAS**: 弹性推理服务（Elastic Algorithm Service）
- **PAI-Blade**: 推理优化引擎
- **灵骏**: 大模型训练/推理平台
- **通义千问**: 大模型推理服务

### Likely Stack（推测）
- **推理框架**: PAI-Blade + vLLM 改造 + 自研
- **Kernel**: 自研 CUDA kernel + Triton
- **硬件**: NVIDIA A100/H100 + 自研芯片（含光）
- **分布式**: 自研 TP/PP + NCCL
- **调度**: K8s + 自研 GPU scheduler
- **量化**: INT8/INT4 自研方案
- **语言**: Python, C++, CUDA, Java

### Target Roles
1. **推理引擎开发工程师** — 推理框架开发
2. **GPU 优化工程师** — CUDA kernel 优化
3. **AI 平台工程师** — PAI 平台开发

### Interview Focus
- 推理系统设计（multi-tenant, autoscaling）
- CUDA 基础（memory hierarchy, kernel optimization）
- vLLM/推理框架内部机制
- 系统设计（大规模 serving）
- 算法题（LeetCode Medium-Hard）

### Resume Angle
- 突出 vLLM 开源贡献
- 突出推理系统理解
- 突出性能优化数据
- 浙大校友网络加分

### Project Evidence Needed
- CUDA 基础 kernel 实现
- 推理 benchmark 报告
- vLLM 深度使用经验
- 系统设计能力

### Risk
- **面试难度**: 中-高
- **竞争**: 中
- **薪资**: 中-高（阿里 P6/P7）
- **成长空间**: 高

### Application Strategy
- **P1 级别（补 CUDA 基础后投递）**
- 通过浙大校友网络内推
- 补完 CUDA softmax/GEMM 后投递
- 中文简历突出推理系统经验

---

## 3. 火山引擎

### 基本信息
- **总部**: 北京
- **规模**: 大型（字节跳动旗下）
- **特点**: 字节跳动 AI Infra 对外输出

### Infra Product
- **方舟**: 大模型推理服务平台
- **AML (Angel Machine Learning)**: ML 平台
- **豆包**: 大模型推理服务
- **自研推理引擎**: 支撑字节内部大规模推理

### Likely Stack（推测）
- **推理框架**: 自研推理引擎（高度优化）
- **Kernel**: 自研 CUDA kernel（极致优化）
- **硬件**: NVIDIA H100/H800 大规模集群
- **分布式**: 自研 TP/PP/EP + NCCL
- **调度**: K8s + 自研 GPU scheduler（TCE）
- **量化**: FP8/INT8 自研
- **Speculative Decoding**: 内部使用
- **语言**: Python, C++, CUDA, Go

### Target Roles
1. **大模型推理系统工程师** — 推理引擎开发
2. **推理优化工程师** — CUDA kernel 优化
3. **AI Infra 工程师** — 平台开发

### Interview Focus
- **CUDA 深度**: kernel 编写, memory optimization
- 推理系统设计（大规模 serving）
- 性能优化方法论
- 分布式推理
- 算法题（LeetCode Medium-Hard）
- 系统设计

### Resume Angle
- 突出 vLLM 开源贡献
- 突出 speculative decoding 经验
- 突出性能优化数据
- 突出推理系统理解

### Project Evidence Needed
- CUDA kernel 实现
- 推理 benchmark 报告
- Profiling 分析报告
- 系统设计能力

### Risk
- **面试难度**: 中-高（CUDA 要求）
- **竞争**: 高（字节吸引力强）
- **薪资**: 高（字节水平）
- **成长空间**: 极高（大规模推理场景）

### Application Strategy
- **P1 级别（补 CUDA 基础后投递）**
- 通过浙大校友网络内推
- 火山引擎是字节 AI Infra 对外窗口
- 补完 CUDA 基础 + profiling 报告后投递

---

## 综合对比

| 公司 | 技术匹配 | 面试难度 | 薪资 | 成长空间 | 推荐优先级 |
|------|----------|----------|------|----------|-----------|
| 华为昇腾 | ⭐⭐⭐⭐⭐ | 中 | 中 | 中-高 | P0 |
| 阿里云 PAI | ⭐⭐⭐⭐ | 中-高 | 中-高 | 高 | P1 |
| 火山引擎 | ⭐⭐⭐⭐ | 中-高 | 高 | 极高 | P1 |

---

## 国内投递策略总结

### 投递顺序
1. **华为昇腾**（本周）— 最匹配，保底
2. **火山引擎**（Week 3-4）— 高薪高成长
3. **阿里云 PAI**（Week 3-4）— 稳定大平台

### 内推渠道
- **华为**: vLLM-Ascend PR reviewer 引荐
- **阿里/火山**: 浙大校友网络
- **通用**: 脉脉/LinkedIn 联系团队成员

### 面试准备重点
- EAGLE-3 STAR 故事（中文版，2 分钟）
- CUDA 基础面试题（memory hierarchy, warp, SM）
- 推理系统设计（KV cache, batching, scheduling）
- 算法题（每天 2-3 题 LeetCode）

### 简历版本
- **中文-推理系统版**: 突出 vLLM + NPU + speculative decoding
- 技术术语保留英文（CUDA, KV cache, speculative decoding）
- 量化数据突出（55% throughput improvement, PR #1032 merged）
