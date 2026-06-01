# 面试准备度最终评分

> 评估日期：2026-06-01
> 基于 8 场模拟面试 + 错题本 + 候选人当前能力

---

## 综合评分

| 方向 | 当前分数 (1-10) | 目标分数 | 差距 | 预计达标时间 |
|------|----------------|---------|------|------------|
| LLM Inference 系统设计 | 6.0 | 7.5 | 1.5 | 4 周 |
| CUDA Kernel 优化 | 2.0 | 6.0 | 4.0 | 8 周 |
| Triton Kernel | 1.5 | 5.5 | 4.0 | 6 周 |
| RAG Infrastructure | 6.0 | 7.0 | 1.0 | 3 周 |
| GPU Cluster / K8s | 2.0 | 5.0 | 3.0 | 10 周 |
| Production Debugging | 4.0 | 6.5 | 2.5 | 6 周 |
| Behavioral / Project | 7.0 | 8.0 | 1.0 | 2 周 |
| System Design（综合） | 4.5 | 7.0 | 2.5 | 8 周 |

**综合面试准备度：4.1 / 10**

---

## 各方向详细评估

### LLM Inference（6.0/10）

**强项：**
- 有 vLLM 实战经验，能讲 EAGLE-3 实现细节
- 理解 Speculative Decoding 原理和 tradeoff
- 有量化数据（吞吐 +55%，TPOT -39%）

**弱项：**
- 缺乏 SGLang/TensorRT-LLM 对比经验
- 无 continuous batching scheduler 深度理解
- 无 production scale serving 经验
- 无量化（INT8/FP8）实操

**面试通过概率：**
- 华为昇腾推理岗：70%
- 国内大模型公司推理岗：45%
- Together AI / Fireworks AI：30%
- NVIDIA TensorRT-LLM：15%

---

### CUDA Kernel（2.0/10）

**强项：**
- 有学习计划和明确路径
- 数学基础扎实（西交数学本科）

**弱项：**
- 零 kernel 编写经验
- 无 Nsight profiling 经验
- 无 memory optimization 实操
- 无法回答任何 CUDA 深度追问

**面试通过概率：**
- 任何 CUDA 岗位：5%（当前）
- 完成 CUDA Lab 后：40-50%

---

### RAG Infrastructure（6.0/10）

**强项：**
- 有完整 RAG 链路独立交付经验
- 有 RAGAS 评测经验
- 理解 metadata filters 和双链路设计

**弱项：**
- 缺乏大规模部署经验（百万级文档、千 QPS）
- 无 Milvus/Weaviate 集群运维经验
- 无 hybrid search 性能对比数据
- 缺乏 reranker 选型和优化经验

**面试通过概率：**
- 中小公司 RAG 后端岗：60%
- 大厂 RAG 平台岗：30%

---

### Behavioral / Project Deep Dive（7.0/10）

**强项：**
- 有真实开源协作故事（PR 合入）
- 有量化成果可讲
- STAR 结构清晰

**弱项：**
- 项目数量少（只有 2 个核心项目）
- 缺乏 failure/incident 故事
- 缺乏团队协作规模化故事

**面试通过概率：**
- Behavioral 轮：65%

---

## 优先练习清单

### 本周必须能回答的 10 个问题

1. vLLM 的 PagedAttention 是怎么工作的？block size 是多少？
2. EAGLE-3 和 EAGLE-2 的核心区别是什么？
3. 你的 PR 改了哪些文件？为什么要改这些文件？
4. KV cache 在推测解码中如何管理？rejected token 的 cache 如何回收？
5. Continuous batching 和 static batching 的区别？
6. 你的 RAG 系统 chunk size 是多少？为什么选这个值？
7. RAGAS 评测具体评估哪些维度？
8. 如果让你重新设计这个 RAG 系统，你会改什么？
9. 你遇到的最难的 bug 是什么？怎么定位的？
10. 为什么选择在 NPU 上做推测解码而不是其他优化方向？

### 第 2-4 周必须能回答的追加问题

11. Prefill 和 Decode 阶段的计算特征有什么区别？
12. 如何估算一个 70B 模型的 KV cache 内存需求？
13. vLLM V1 和 V0 的 scheduler 有什么区别？
14. 如果 TTFT 突然升高，你会怎么排查？
15. FlashAttention 的核心思想是什么？IO 复杂度是多少？
16. 什么是 Roofline Model？如何判断一个 kernel 是 compute-bound 还是 memory-bound？
17. Tensor Parallelism 和 Pipeline Parallelism 的区别？各自适合什么场景？
18. 如何设计一个支持多租户的 LLM serving 系统？
19. 你的 RAG 系统如果要支持 100 万文档，架构需要怎么改？
20. 如何评估一个 reranker 的 ROI（延迟增加 vs 质量提升）？

---

## 面试准备度提升路径

```
当前: 4.1/10
  ↓ 第 4 周末（CUDA 基础完成）
中期: 5.5/10
  ↓ 第 8 周末（Benchmark + RAG Eval 完成）
目标: 6.8/10
  ↓ 第 12 周末（Mock Interview 密集练习后）
冲刺: 7.2/10
```

**注意：** 7.0+ 是大多数 AI Infra 岗位的面试通过线。当前 4.1 意味着需要 8-10 周的密集准备才能达到投递标准。
