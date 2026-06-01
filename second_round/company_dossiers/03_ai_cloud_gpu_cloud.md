# AI Cloud / GPU Cloud 技术画像

---

## 1. CoreWeave

### 基本信息
- **总部**: Roseland, NJ / NYC
- **规模**: 1000+ 员工
- **估值**: $35B+（推测）
- **特点**: GPU-native cloud，专注 AI workload

### Infra Product
- GPU 云计算平台（H100/B200 集群）
- Kubernetes-native GPU orchestration
- Inference-as-a-Service
- 模型训练平台

### Likely Stack（推测）
- **编排**: Kubernetes + 自研 GPU scheduler
- **推理**: vLLM / TensorRT-LLM / 自研 serving layer
- **网络**: InfiniBand / RoCE
- **存储**: 高性能分布式存储（Ceph/自研）
- **监控**: Prometheus + Grafana + 自研 GPU metrics
- **语言**: Go, Python, C++
- **硬件**: NVIDIA H100/B200, 大规模集群

### Target Roles
1. **Software Engineer, Inference AI/ML** — 推理系统开发
2. **Senior SWE - Performance and Benchmarking** — 性能工程
3. **Infrastructure Engineer** — 基础设施开发

### Interview Focus
- Kubernetes GPU scheduling
- 推理系统设计（multi-tenant serving）
- 性能 benchmark 方法论
- 分布式系统（集群管理）
- Coding（系统编程风格）

### Resume Angle
- 突出 vLLM 开源贡献（CoreWeave 使用 vLLM）
- 突出性能优化数据（55% throughput improvement）
- 突出 benchmark 设计能力

### Project Evidence Needed
- LLM inference benchmark 报告
- vLLM 部署和调优经验
- K8s GPU serving 基础理解

### Risk
- **面试难度**: 中-高
- **签证**: H1B 支持
- **竞争**: 中等

### Application Strategy
- **P0 — 立即投递**
- vLLM 经验直接对口
- 通过 vLLM 社区网络内推
- 准备 inference benchmark 数据

---

## 2. Lambda

### 基本信息
- **总部**: San Francisco, CA
- **规模**: 300+ 员工
- **特点**: GPU 云 + 深度学习工作站

### Infra Product
- Lambda Cloud（GPU 云实例）
- Lambda 工作站/服务器
- 1-Click Cluster（GPU 集群部署）

### Likely Stack（推测）
- **编排**: Kubernetes + Slurm
- **虚拟化**: 裸金属 + 容器
- **网络**: InfiniBand
- **监控**: 标准 observability stack
- **语言**: Python, Go, C++

### Target Roles
1. **Infrastructure Engineer** — 云基础设施
2. **ML Platform Engineer** — ML 平台开发

### Interview Focus
- 云基础设施设计
- GPU 集群管理
- Linux 系统编程
- 网络和存储

### Resume Angle
- 突出系统工程能力
- 突出 GPU 相关经验

### Project Evidence Needed
- GPU 集群部署经验
- 系统性能优化

### Risk
- **面试难度**: 中
- **方向偏差**: 偏 infra，非推理优化
- **签证**: H1B 支持

### Application Strategy
- **P2 级别**
- 偏基础设施方向，与候选人推理优化方向有偏差
- 作为备选

---

## 3. NVIDIA

### 基本信息
- **总部**: Santa Clara, CA
- **规模**: 30000+ 员工
- **市值**: $3T+

### Infra Product
- GPU 硬件（H100/B200/GB200）
- CUDA / cuDNN / NCCL
- TensorRT / TensorRT-LLM
- Triton Inference Server
- NeMo Framework
- NVIDIA Dynamo

### Likely Stack
- **语言**: C++, CUDA, Python
- **推理**: TensorRT-LLM + Triton Inference Server
- **编译器**: NVCC, Triton, CUTLASS
- **Profiling**: Nsight Systems / Nsight Compute
- **分布式**: NCCL, NVLink, NVSwitch

### Target Roles
1. **SWE - ML Inference (New Grad)** — 推理系统开发
2. **AI Inference Performance Engineer** — 性能工程
3. **Senior GenAI Algorithms Engineer** — 算法优化
4. **Senior DL Frameworks CUDA SWE** — CUDA 框架开发

### Interview Focus
- **CUDA 深度**: 手写 kernel, memory hierarchy, warp-level programming
- **性能优化**: Nsight profiling, roofline analysis
- **TensorRT**: plugin 开发, 模型优化
- **系统设计**: 推理系统架构
- **Coding**: LeetCode Medium-Hard

### Resume Angle
- 突出 CUDA kernel 学习项目
- 突出 GPU profiling 报告
- 突出 TensorRT-LLM 使用经验
- 突出 Ascend 经验（展示跨硬件理解）

### Project Evidence Needed
- CUDA kernel 实现（GEMM, attention, softmax）
- Nsight Compute profiling 报告
- TensorRT-LLM 部署和 benchmark
- 性能优化案例（before → after）

### Risk
- **面试难度**: 高（CUDA 深度要求）
- **竞争**: 高
- **签证**: H1B/L1 支持

### Application Strategy
- **P2 级别（2-3 个月后）**
- New Grad 岗位门槛相对较低
- 需要完成 CUDA kernel 学习项目
- 通过 TensorRT-LLM 开源贡献建立联系

---

## 4. AMD

### 基本信息
- **总部**: Santa Clara, CA
- **规模**: 25000+ 员工
- **特点**: MI300X GPU，ROCm 生态

### Infra Product
- MI300X/MI400 GPU
- ROCm（开源 GPU 计算平台）
- Composable Kernel
- hipBLAS / MIOpen

### Likely Stack
- **语言**: C++, HIP (类 CUDA), Python
- **编译器**: ROCm, hipcc
- **推理**: vLLM ROCm backend / TGI
- **通信**: RCCL（ROCm 版 NCCL）

### Target Roles
1. **ML Inference Engineer** — 推理优化
2. **ROCm Software Engineer** — GPU 软件栈

### Interview Focus
- GPU 编程（HIP/CUDA）
- 性能优化
- 编译器/runtime 知识
- 跨平台适配

### Resume Angle
- **突出 Ascend NPU 经验** — 非 NVIDIA 硬件适配是核心加分项
- 突出跨平台推理框架经验

### Project Evidence Needed
- 非 NVIDIA 硬件适配经验
- GPU kernel 编写
- 性能 benchmark

### Risk
- **面试难度**: 中-高
- **竞争**: 中（比 NVIDIA 低）
- **优势**: Ascend 经验是差异化优势

### Application Strategy
- **P2 级别**
- Ascend 经验是独特优势（非 NVIDIA 硬件适配）
- OpenAI AMD GPU 岗位也可考虑

---

## 5. AWS

### 基本信息
- **总部**: Seattle, WA
- **规模**: 大型

### Infra Product
- SageMaker（ML 平台）
- Inferentia / Trainium（自研 AI 芯片）
- Neuron SDK
- Bedrock（模型 API 服务）

### Likely Stack
- **推理**: Neuron SDK + 自研 runtime
- **硬件**: Inferentia2 / Trainium2 + NVIDIA GPU
- **编排**: EKS (K8s) + 自研
- **语言**: Python, C++, Java

### Target Roles
1. **ML Inference Engineer** — 推理系统
2. **Neuron SDK Engineer** — 自研芯片软件栈

### Interview Focus
- 系统设计（Amazon 风格 LP）
- 推理系统知识
- 分布式系统
- Coding

### Resume Angle
- 突出非 NVIDIA 硬件经验（Ascend → Inferentia 迁移能力）
- 突出推理框架经验

### Project Evidence Needed
- 推理系统设计
- 非 NVIDIA 硬件适配

### Risk
- **面试难度**: 中-高（LP 面试）
- **签证**: H1B 支持

### Application Strategy
- **P2 级别**
- Neuron SDK 方向与 Ascend 经验有迁移价值

---

## 6. GCP (Google Cloud)

### 基本信息
- **总部**: Mountain View, CA

### Infra Product
- Vertex AI（ML 平台）
- TPU（自研 AI 芯片）
- Cloud GPU（NVIDIA A100/H100）

### Likely Stack
- **推理**: JAX/XLA + 自研
- **硬件**: TPU v4/v5 + NVIDIA GPU
- **编排**: GKE (K8s)
- **语言**: Python, C++, Go

### Target Roles
1. **ML Infrastructure Engineer** — ML 基础设施
2. **TPU Performance Engineer** — TPU 性能优化

### Interview Focus
- 系统设计（Google 风格）
- Coding（LeetCode Medium-Hard）
- ML 系统知识

### Resume Angle
- 突出系统设计能力
- 突出性能优化方法论

### Risk
- **面试难度**: 高
- **门槛**: Google 标准面试流程

### Application Strategy
- **P2-P3 级别**
- 通过 Google 校友网络

---

## 7. Azure (Microsoft)

### 基本信息
- **总部**: Redmond, WA

### Infra Product
- Azure AI / Azure OpenAI Service
- Azure Machine Learning
- ONNX Runtime
- Azure Maia（自研 AI 芯片）

### Likely Stack
- **推理**: ONNX Runtime + TensorRT-LLM
- **硬件**: NVIDIA GPU + AMD MI300X + Maia
- **编排**: AKS (K8s)
- **语言**: C++, Python, C#

### Target Roles
1. **SWE - Azure AI Inference** — 推理系统
2. **ONNX Runtime Engineer** — 推理引擎

### Interview Focus
- 系统设计
- Coding
- 推理优化
- ONNX 知识

### Resume Angle
- 突出跨硬件适配经验
- 突出推理框架经验

### Risk
- **面试难度**: 中-高
- **签证**: H1B 支持

### Application Strategy
- **P2 级别**
- 见 Frontier Labs 中 Microsoft AI 部分
