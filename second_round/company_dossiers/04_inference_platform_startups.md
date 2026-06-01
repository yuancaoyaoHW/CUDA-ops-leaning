# Inference Platform Startups 技术画像

---

## 1. Together AI

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 200+ 员工（推测）
- **融资**: $400M+（推测）
- **特点**: 开源模型推理平台，vLLM/SGLang 核心贡献者

### Infra Product
- Together Inference（高性能推理 API）
- Together Fine-tuning
- Together Embeddings
- 开源贡献：FlashAttention (Tri Dao), SGLang

### Likely Stack（推测）
- **推理框架**: vLLM + SGLang + 自研优化层
- **Kernel**: FlashAttention + FlashInfer + Triton kernel
- **硬件**: NVIDIA H100 集群
- **分布式**: TP/PP，自研调度
- **量化**: FP8 / INT8 / AWQ
- **Speculative Decoding**: 深度使用（Snowflake Arctic 合作）
- **语言**: Python, C++, CUDA, Rust

### Target Roles
1. **Inference Engineer** — 推理系统优化
2. **Kernel Engineer** — CUDA/Triton kernel 开发
3. **ML Infrastructure Engineer** — ML 平台

### Interview Focus
- 推理系统设计（continuous batching, KV cache management）
- CUDA kernel 基础（不需要极深，但需要理解）
- 性能优化方法论（profiling, bottleneck analysis）
- vLLM/SGLang 内部机制
- Coding（Python + 系统设计）

### Resume Angle
- **核心切入**: vLLM 开源贡献 + speculative decoding 经验
- Together AI 是 vLLM/SGLang 生态的核心参与者
- 突出 EAGLE-3 经验（Together AI 关注 spec decode）
- 突出性能优化数据

### Project Evidence Needed
- vLLM/SGLang 深度使用和贡献
- CUDA 基础 benchmark
- Speculative decoding 实践
- 推理性能 profiling 报告

### Risk
- **面试难度**: 中-高
- **竞争**: 高（热门公司）
- **签证**: H1B 支持

### Application Strategy
- **P1 级别（补 CUDA benchmark 后投递）**
- 通过 vLLM/SGLang 社区网络内推
- 突出 spec decode 经验（Together AI 与 Snowflake 合作 spec decode）
- 补完 CUDA 基础 benchmark 后投递

---

## 2. Fireworks AI

### 基本信息
- **总部**: Redwood City, CA
- **规模**: 100+ 员工（推测）
- **特点**: 高性能推理 API，极致延迟优化

### Infra Product
- Fireworks Inference API（极低延迟）
- FireFunction（function calling 优化）
- 模型微调服务
- Compound AI 系统

### Likely Stack（推测）
- **推理框架**: 自研推理引擎（非纯 vLLM）
- **Kernel**: 自研 CUDA kernel + Triton
- **硬件**: NVIDIA H100
- **优化**: 极致延迟优化（kernel fusion, memory optimization）
- **量化**: FP8 / INT4 aggressive quantization
- **语言**: C++, CUDA, Python, Rust

### Target Roles
1. **SWE - AI Infrastructure** — AI 基础设施
2. **SWE - Performance Optimization** — 性能优化

### Interview Focus
- **性能优化深度**: profiling → bottleneck → optimization cycle
- CUDA kernel 理解（不一定手写，但需要理解）
- 推理系统设计
- 延迟优化技巧（kernel fusion, memory layout）
- Coding

### Resume Angle
- 突出性能优化经验（55% throughput improvement）
- 突出 profiling 方法论
- 突出推理系统理解

### Project Evidence Needed
- GPU profiling 报告（Nsight）
- 性能优化案例（before → after with data）
- CUDA 基础理解
- 推理 benchmark

### Risk
- **面试难度**: 中-高
- **竞争**: 中
- **签证**: H1B 支持

### Application Strategy
- **P1 级别（补 GPU profiling 后投递）**
- 产出一份完整的 Nsight profiling 报告
- 突出性能优化方法论

---

## 3. Baseten

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 100+ 员工（推测）
- **特点**: 模型部署平台，Truss 框架

### Infra Product
- Baseten Platform（模型部署）
- Truss（开源模型打包框架）
- Autoscaling GPU inference
- Multi-model serving

### Likely Stack（推测）
- **推理框架**: vLLM + TensorRT-LLM + 自研 serving layer
- **编排**: Kubernetes + 自研 autoscaler
- **打包**: Truss（Docker-based）
- **语言**: Python, Go
- **监控**: 自研 metrics + Prometheus

### Target Roles
1. **Applied AI Inference Engineer** — 推理系统应用
2. **Forward Deployed Engineer** — 客户对接

### Interview Focus
- 推理系统设计（multi-tenant, autoscaling）
- Python 工程能力
- 模型部署流程
- 系统设计（serving platform）
- Customer-facing 能力（FDE 岗位）

### Resume Angle
- 突出 vLLM 使用和贡献经验
- 突出推理系统理解
- 突出 Python 工程能力

### Project Evidence Needed
- 模型部署经验
- 推理系统设计
- Python 工程项目

### Risk
- **面试难度**: 中
- **门槛**: 1+ years（较低）
- **签证**: H1B 支持

### Application Strategy
- **P0 — 立即投递**
- 门槛较低，匹配度高
- 突出 vLLM 经验

---

## 4. Replicate

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 100+ 员工（推测）
- **特点**: 模型托管/推理平台，Cog 框架

### Infra Product
- Replicate Platform（模型托管）
- Cog（开源模型打包）
- Streaming inference
- 多模态模型支持

### Likely Stack（推测）
- **推理框架**: 多框架支持（vLLM, TGI, 自研）
- **编排**: Kubernetes + 自研
- **打包**: Cog（Docker-based）
- **语言**: Python, Go
- **硬件**: NVIDIA GPU（多种型号）

### Target Roles
1. **ML Infrastructure Engineer** — ML 基础设施
2. **Backend Engineer** — 后端开发

### Interview Focus
- 系统设计（model serving platform）
- Python/Go 工程
- 容器化和编排
- 推理系统知识

### Resume Angle
- 突出推理系统经验
- 突出 Python 工程能力
- 突出模型部署经验

### Project Evidence Needed
- 模型部署和 serving 经验
- 系统设计能力
- Python 工程项目

### Risk
- **面试难度**: 中
- **竞争**: 中
- **签证**: H1B 支持

### Application Strategy
- **P1 级别**
- 补充 model serving 部署经验后投递

---

## 5. Anyscale

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 300+ 员工（推测）
- **特点**: Ray 框架商业化，分布式计算

### Infra Product
- Anyscale Platform（Ray 托管平台）
- Ray Serve（模型 serving）
- Ray Train（分布式训练）
- vLLM on Ray 集成

### Likely Stack
- **框架**: Ray（核心）
- **推理**: Ray Serve + vLLM
- **分布式**: Ray 分布式计算
- **编排**: Kubernetes + Ray Cluster
- **语言**: Python, C++
- **硬件**: 多云 GPU

### Target Roles
1. **Distributed LLM Inference Engineer** — 分布式推理
2. **Ray Serve Engineer** — serving 框架开发

### Interview Focus
- 分布式系统设计
- Ray 框架理解
- LLM 推理系统
- Python 工程
- Coding

### Resume Angle
- 突出推理系统经验
- 突出分布式理解（即使是理论层面）
- 突出 vLLM 经验（Anyscale 集成 vLLM）

### Project Evidence Needed
- 分布式推理实践
- Ray 使用经验
- vLLM 多卡部署

### Risk
- **面试难度**: 中-高
- **竞争**: 中
- **签证**: H1B 支持

### Application Strategy
- **P1 级别**
- 补充分布式推理实践后投递
- 学习 Ray Serve 基础

---

## 6. Modal

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 50+ 员工（推测）
- **特点**: Serverless GPU 计算平台

### Infra Product
- Modal Platform（Serverless GPU）
- 容器化 GPU 函数
- 自动扩缩容
- GPU 调度优化

### Likely Stack（推测）
- **编排**: 自研 GPU scheduler（非标准 K8s）
- **容器**: gVisor / Firecracker（推测）
- **语言**: Python, Rust, Go
- **硬件**: NVIDIA GPU（多种型号）
- **特色**: 极快冷启动

### Target Roles
1. **Infrastructure Engineer** — 基础设施开发
2. **GPU Systems Engineer** — GPU 系统

### Interview Focus
- 系统编程（Linux, containers, scheduling）
- GPU 调度和资源管理
- 性能优化
- Rust/Go 编程

### Resume Angle
- 突出系统工程能力
- 突出 GPU 相关经验

### Project Evidence Needed
- 系统编程经验
- GPU 调度理解
- 容器化经验

### Risk
- **面试难度**: 中-高
- **方向偏差**: 偏 infra/systems，非推理优化
- **签证**: H1B 支持

### Application Strategy
- **P2 级别**
- 偏系统方向，与候选人推理优化方向有偏差
- 需要补充系统编程经验

---

## 综合对比

| 公司 | 推理深度 | 入门门槛 | 签证 | 候选人匹配 | 推荐优先级 |
|------|----------|----------|------|-----------|-----------|
| Together AI | ⭐⭐⭐⭐⭐ | 中-高 | ✅ | 高 | P1 |
| Fireworks AI | ⭐⭐⭐⭐⭐ | 中-高 | ✅ | 中-高 | P1 |
| Baseten | ⭐⭐⭐⭐ | 中 | ✅ | 高 | P0 |
| Replicate | ⭐⭐⭐⭐ | 中 | ✅ | 中-高 | P1 |
| Anyscale | ⭐⭐⭐⭐⭐ | 中-高 | ✅ | 中-高 | P1 |
| Modal | ⭐⭐⭐ | 高 | ✅ | 中 | P2 |
