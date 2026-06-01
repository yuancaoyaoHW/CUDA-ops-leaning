# 简历优势与短板分析

## 优势

### 1. 推理系统实战经验（高竞争力）
- 直接参与 vLLM 生态核心功能开发（Speculative Decoding），非 toy project
- 有端到端 benchmark 验证能力，能用数据说话（吞吐、TPOT、acceptance rate）
- 理解推理系统关键路径：KV cache、decode 调度、runner 执行链路

### 2. 开源贡献记录（差异化优势）
- PR 合入主流推理框架（vLLM-Ascend），证明代码质量经过社区 review
- 经历完整开源协作流程：接口适配、reviewer 反馈、测试补充、冲突处理
- 在 AI Infra 求职中，开源贡献是强信号

### 3. 硬件适配经验（稀缺性）
- Ascend NPU 适配经验在华为生态岗位中高度匹配
- 理解异构硬件上的推理优化挑战（非纯 GPU 背景，但有硬件感知）

### 4. 教育背景扎实
- 浙大计算机硕士 + 西交数学本科
- 数学背景对理解 attention 计算、数值精度、量化有潜在优势

### 5. RAG 工程闭环能力
- 独立负责完整 RAG 系统，从文档解析到评测
- 有工程化思维：日志、异常处理、接口抽象

---

## 短板 / Gap 分析

### 🔴 关键 Gap（多数 AI Infra 岗位硬性要求）

| Gap | 说明 | 影响范围 |
|-----|------|----------|
| **CUDA kernel 开发** | 简历无 CUDA 编程经验，无自定义 kernel 实现 | GPU 推理优化岗、算子开发岗直接卡门槛 |
| **GPU profiling** | 无 Nsight、NCU、nvprof 等工具使用记录 | 性能优化岗需要 profiling 驱动优化的能力 |
| **Triton 编程** | 无 Triton kernel 编写经验 | 越来越多团队用 Triton 替代手写 CUDA |
| **分布式推理** | 无 Tensor Parallelism / Pipeline Parallelism / 多卡推理经验 | 大规模部署岗位核心要求 |

### 🟡 重要 Gap（中高级岗位加分项）

| Gap | 说明 | 影响范围 |
|-----|------|----------|
| **量化技术** | 无 INT8/INT4/FP8 量化、AWQ、GPTQ 等经验 | 推理优化岗常见要求 |
| **TensorRT / TensorRT-LLM** | 无 NVIDIA 推理引擎使用经验 | NVIDIA 生态岗位核心技能 |
| **NCCL / 通信优化** | 无分布式通信库经验 | 分布式推理、训练框架岗位 |
| **GPU 架构理解** | 无 SM、warp、shared memory、bank conflict 等底层优化记录 | 算子优化岗面试高频考点 |

### 🟢 次要 Gap（可后续补充）

| Gap | 说明 |
|-----|------|
| Continuous Batching 深度 | 有 decode 调度经验但未体现 batching 策略设计 |
| Model Serving 系统设计 | 无 serving 框架（Triton Inference Server）部署经验 |
| 训练侧经验 | 无分布式训练、FSDP、DeepSpeed 经验 |

---

## 补强建议（按优先级排序）

### P0：立即补强（1-2 周可出成果）

1. **CUDA kernel 基础**
   - 实现 softmax、GEMM、reduce 等基础 kernel
   - 用 NCU 做 profiling 并写优化报告
   - 目标：能在面试中手写简单 kernel + 解释优化思路

2. **GPU profiling 实操**
   - 用 Nsight Compute 分析现有 PyTorch 模型的 kernel 性能
   - 产出 profiling 报告，体现 memory bound vs compute bound 分析

### P1：重点补强（2-4 周）

3. **Triton kernel 编写**
   - 用 Triton 实现 FlashAttention 简化版或 fused kernel
   - 与 PyTorch native 实现做性能对比

4. **分布式推理理解**
   - 阅读 vLLM 的 TP/PP 实现代码
   - 能在面试中讲清 Tensor Parallelism 的通信模式

### P2：加分项（有余力时补充）

5. **量化实践**：用 AWQ/GPTQ 量化一个模型并做 benchmark
6. **TensorRT-LLM 体验**：跑通一个模型的 TRT-LLM 部署流程
