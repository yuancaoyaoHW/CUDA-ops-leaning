# Frontier AI Labs 技术画像

---

## 1. OpenAI

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 3000+ 员工
- **估值**: $300B+（推测）

### Infra Product
- GPT-4/5 系列模型推理服务
- ChatGPT / API 平台
- 内部推理引擎（非开源）
- Triton Inference Server（早期使用，现已自研）

### Likely Stack（推测）
- **推理框架**: 自研推理引擎（非 vLLM/TRT-LLM）
- **Kernel**: 自研 CUDA kernel + Triton kernel
- **硬件**: NVIDIA H100/B200 集群 + AMD MI300X（新方向）
- **分布式**: 自研 TP/PP 实现，NCCL/自研通信
- **调度**: Kubernetes + 自研 GPU scheduler
- **量化**: FP8 / INT8 自研方案
- **Speculative Decoding**: 内部实现（推测）
- **监控**: Prometheus + Grafana + 自研 observability

### Target Roles
1. **SWE - Model Inference** — CUDA kernel + 推理系统优化
2. **SWE - Inference AMD GPU** — 非 NVIDIA 硬件适配（候选人 Ascend 经验加分）
3. **SWE - Inference Infrastructure** — 推理平台基础设施

### Interview Focus
- CUDA kernel 编写（手写 attention/GEMM）
- 系统设计（设计一个 serving system for 1B+ requests/day）
- 分布式推理（TP/PP 实现细节）
- 性能优化方法论（profiling → bottleneck → optimization）
- Coding（LeetCode Hard 级别）

### Resume Angle
- 突出 vLLM 开源贡献 + speculative decoding 经验
- 突出 Ascend NPU 经验（对 AMD GPU 岗位有迁移价值）
- 突出性能优化数据（55% throughput improvement）

### Project Evidence Needed
- CUDA kernel 实现（GEMM, FlashAttention 级别）
- 分布式推理实践（TP=2/4 部署）
- 完整的 profiling 报告
- 系统设计文档

### Risk
- **面试难度**: 极高（LeetCode Hard + 系统设计 + CUDA 深度）
- **竞争**: 极激烈
- **签证**: H1B 支持但名额有限

### Application Strategy
- 通过 vLLM 社区 contributor 网络内推
- AMD GPU 岗位是差异化切入点（Ascend 经验 → 非 NVIDIA 硬件）
- 需要 2-3 个月 CUDA 深度准备

---

## 2. Anthropic

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 1500+ 员工
- **估值**: $60B+（推测）

### Infra Product
- Claude 系列模型推理服务
- Claude API 平台
- 内部推理/训练基础设施

### Likely Stack（推测）
- **推理框架**: 自研（推测基于 JAX/XLA）
- **硬件**: Google TPU + NVIDIA GPU 混合
- **分布式**: JAX pjit / XLA sharding
- **调度**: Kubernetes + 自研 orchestration
- **语言**: Python, Rust, C++
- **监控**: 自研 observability

### Target Roles
1. **Staff Infra Engineer, Cluster** — GPU/TPU 集群管理
2. **ML Infrastructure Engineer** — 训练/推理基础设施

### Interview Focus
- 分布式系统设计（大规模集群）
- Kubernetes 深度（GPU scheduling, resource management）
- 系统可靠性（SRE 思维）
- Coding（系统编程风格）

### Resume Angle
- 突出系统工程能力
- 突出开源贡献展示的协作能力
- 不适合突出 CUDA kernel（Anthropic 偏 infra）

### Project Evidence Needed
- K8s GPU serving 经验
- 大规模系统运维经验
- 分布式系统设计

### Risk
- **面试难度**: 极高
- **门槛**: 7+ years 经验要求
- **方向偏差**: Anthropic 偏 cluster infra，非推理优化

### Application Strategy
- 长期目标（P3）
- 需要积累 K8s + 集群管理经验
- 当前不建议投递

---

## 3. Google DeepMind

### 基本信息
- **总部**: London, UK / Mountain View, CA
- **规模**: 3000+ 员工

### Infra Product
- Gemini 系列模型
- Google Cloud Vertex AI
- TPU 推理优化

### Likely Stack（推测）
- **推理框架**: 自研（基于 JAX/XLA/Pathways）
- **硬件**: TPU v4/v5 为主，部分 GPU
- **分布式**: Pathways / JAX pjit
- **语言**: Python, C++, JAX
- **调度**: Borg（Google 内部）

### Target Roles
1. **Staff ML Engineer, Inference** — 推理系统优化
2. **SWE, ML Infrastructure** — ML 基础设施

### Interview Focus
- 系统设计（Google 风格，大规模）
- Coding（LeetCode Medium-Hard）
- ML 系统知识
- TPU/XLA 理解（加分）

### Resume Angle
- 突出系统设计能力
- 突出性能优化方法论
- Google 看重学术背景（浙大计算机硕士加分）

### Project Evidence Needed
- 大规模系统设计经验
- 性能优化案例
- 学术论文或技术博客

### Risk
- **面试难度**: 极高
- **门槛**: 7+ years 或 PhD
- **签证**: L1/H1B 支持

### Application Strategy
- 长期目标（P3）
- 通过 Google 校友网络
- 需要 6+ 个月积累

---

## 4. Microsoft AI

### 基本信息
- **总部**: Redmond, WA
- **规模**: 大型（AI 部门数千人）

### Infra Product
- Azure AI / Azure OpenAI Service
- ONNX Runtime
- DeepSpeed
- Phi 系列模型

### Likely Stack（推测）
- **推理框架**: ONNX Runtime + TensorRT-LLM + 自研
- **硬件**: NVIDIA GPU + AMD MI300X + 自研 Maia
- **分布式**: DeepSpeed / ONNX Runtime 分布式
- **调度**: Azure Kubernetes Service
- **语言**: C++, Python, C#

### Target Roles
1. **SWE - Inference** — 推理系统优化
2. **SWE - ONNX Runtime** — 推理引擎开发

### Interview Focus
- 系统设计
- Coding（LeetCode Medium）
- ONNX/推理优化知识
- 分布式系统

### Resume Angle
- 突出推理框架经验（vLLM → ONNX Runtime 迁移能力）
- 突出跨硬件适配经验（Ascend → AMD/Maia）

### Project Evidence Needed
- ONNX 模型优化经验
- 推理性能 benchmark
- 分布式推理实践

### Risk
- **面试难度**: 中-高
- **竞争**: 中等
- **签证**: H1B 支持

### Application Strategy
- P2 级别（2-3 个月后）
- 通过 DeepSpeed 开源贡献建立联系
- ONNX Runtime 方向门槛相对较低

---

## 5. DeepSeek

### 基本信息
- **总部**: 北京/杭州
- **规模**: 500+ 员工（推测）
- **特点**: 技术驱动，开源友好

### Infra Product
- DeepSeek-V3/V4 系列模型
- 自研推理引擎
- MoE 推理优化

### Likely Stack（推测）
- **推理框架**: 自研（基于 vLLM 改造，推测）
- **Kernel**: 自研 CUDA kernel（MoE routing, attention）
- **硬件**: NVIDIA H100/H800 集群
- **分布式**: 自研 TP/PP/EP（Expert Parallelism）
- **量化**: FP8 / INT8 自研
- **语言**: Python, C++, CUDA

### Target Roles
1. **推理系统工程师** — 推理引擎开发
2. **CUDA 优化工程师** — kernel 开发

### Interview Focus
- CUDA kernel 编写（手写 attention/GEMM）
- MoE 推理优化
- 分布式推理（TP/PP/EP）
- 系统设计
- 算法题

### Resume Angle
- 突出 vLLM 开源贡献
- 突出 speculative decoding 经验
- 突出性能优化数据

### Project Evidence Needed
- CUDA kernel 实现
- MoE 相关理解
- 分布式推理实践

### Risk
- **面试难度**: 高（CUDA 深度要求）
- **竞争**: 高（技术氛围好，吸引力强）
- **薪资**: 有竞争力（推测）

### Application Strategy
- P2 级别（2-3 个月后）
- 补完 CUDA 基础后投递
- 通过开源社区建立联系

---

## 6. Moonshot (月之暗面)

### 基本信息
- **总部**: 北京
- **规模**: 500+ 员工（推测）
- **产品**: Kimi

### Infra Product
- Kimi 长上下文推理
- 自研推理引擎
- 长序列优化

### Likely Stack（推测）
- **推理框架**: 自研
- **Kernel**: 自研 CUDA kernel（长序列 attention）
- **硬件**: NVIDIA H100/H800
- **特色**: 长上下文优化（128K+ tokens）
- **语言**: Python, C++, CUDA

### Target Roles
1. **推理引擎工程师** — 推理系统开发
2. **CUDA 优化工程师** — kernel 开发

### Interview Focus
- 长序列 attention 优化
- CUDA kernel 编写
- KV cache 管理（长上下文场景）
- 系统设计

### Resume Angle
- 突出 attention 相关经验
- 突出 KV cache 管理理解
- 突出 speculative decoding（长序列场景价值高）

### Project Evidence Needed
- CUDA attention kernel
- 长序列推理 benchmark
- KV cache 优化方案

### Risk
- **面试难度**: 高
- **竞争**: 中-高

### Application Strategy
- P2 级别（2-3 个月后）
- 长序列优化是差异化方向

---

## 7. MiniMax

### 基本信息
- **总部**: 上海
- **规模**: 500+ 员工（推测）
- **产品**: MiniMax-M2, 海螺 AI

### Infra Product
- MiniMax 系列模型推理
- 多模态推理
- 自研推理引擎

### Likely Stack（推测）
- **推理框架**: 自研（推测）
- **硬件**: NVIDIA GPU 集群
- **语言**: Python, C++, CUDA

### Target Roles
1. **推理优化工程师** — 推理性能优化

### Interview Focus
- CUDA kernel 编写
- 推理系统设计
- 量化技术
- 多模态推理

### Resume Angle
- 突出推理优化经验
- 突出开源贡献

### Project Evidence Needed
- CUDA kernel 实现
- 推理 benchmark

### Risk
- **面试难度**: 中-高
- **竞争**: 中

### Application Strategy
- P2 级别
- 门槛相对 DeepSeek/Moonshot 略低
