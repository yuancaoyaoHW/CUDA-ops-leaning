# 高压追问库 (High-Pressure Follow-up Bank)

候选人背景：浙大硕士+西交本科，vLLM-Ascend PR#1032（EAGLE-3），RAG后端（RAGAS 90%），零CUDA/零Production/零分布式推理

---

## 一、技术深度追问（30 条）

1. 你说 EAGLE-3 吞吐提升 55%，那 acceptance rate 具体是多少？不同 temperature 下 acceptance rate 怎么变化？
2. EAGLE-3 的 draft tree 结构是怎么确定的？tree width 和 depth 的 tradeoff 你做过实验吗？
3. PagedAttention 的 block size 为什么通常选 16？你试过其他值吗？对 fragmentation 的影响是什么？
4. FlashAttention 的 online softmax 具体怎么实现的？如果 tile 之间的 max 不同怎么修正？
5. KV cache 量化到 FP8 会对生成质量有什么影响？你怎么评估这个 degradation？
6. Continuous batching 中 preemption 的 swap 和 recompute 策略，具体什么条件下选哪个？
7. 你提到 Tensor Parallelism 每层需要 2 次 AllReduce，具体是在哪两个位置？通信量怎么算？
8. Speculative decoding 中 verification 的 rejection sampling 具体怎么做？为什么能保证输出分布不变？
9. vLLM 的 scheduler 在 prefill 和 decode 混合调度时，优先级是怎么定的？为什么？
10. 你说 prefix caching 用 hash(token_ids) 做 key，如果两个不同 prompt 的 hash 冲突怎么办？
11. Grouped-Query Attention 相比 Multi-Head Attention 节省多少 KV cache？对模型质量的影响？
12. CUDA kernel 中 warp shuffle 的 `__shfl_down_sync` 的 mask 参数具体怎么用？全 warp 参与和部分参与有什么区别？
13. Shared memory 的 bank conflict 在 4-byte 和 8-byte 访问模式下有什么不同？
14. 你说 roofline model 的 ridge point 是 156 FLOPs/byte，这个数字怎么算出来的？换成 H100 是多少？
15. Tensor Core 的 WMMA API 和 MMA PTX 指令有什么区别？什么时候用哪个？
16. CUDA Graph 在什么情况下不能用？动态 shape 怎么处理？
17. 你的 RAG 系统 reranker 用的什么模型？cross-encoder 的推理延迟具体多少？
18. 向量数据库的 HNSW 索引，ef_construction 和 M 参数怎么调？你做过 ablation 吗？
19. 你说 RAGAS faithfulness 0.90，剩下 10% 的 failure case 是什么类型？怎么改进？
20. BM25 和 dense retrieval 在什么类型的 query 上各自更强？你有数据支撑吗？
21. FlashAttention 在 causal mask 场景下怎么处理？和 non-causal 的实现有什么区别？
22. vLLM 的 block manager 在 beam search 场景下 COW 具体怎么实现？reference count 怎么管理？
23. 你说 NPU 算子融合减少 kernel launch 开销，具体融合了哪些算子？融合前后延迟差多少？
24. INT8 量化的 per-channel 和 per-tensor 有什么区别？为什么 activation 通常用 per-tensor？
25. Speculative decoding 和 continuous batching 结合时，draft tokens 怎么处理 batch 中其他请求的 decode？
26. 你了解 Ring Attention 吗？它和 Tensor Parallelism 的 sequence parallel 有什么区别？
27. vLLM 中 chunked prefill 的 chunk size 怎么选？太小和太大分别有什么问题？
28. GPU 的 L2 cache 在 LLM 推理中起什么作用？怎么利用 L2 cache locality？
29. 你说 EAGLE-3 的 draft overhead 约 15%，这个是怎么测量的？包含哪些开销？
30. Ascend NPU 和 NVIDIA GPU 在算子实现上最大的区别是什么？你迁移时遇到的最大挑战？

---

## 二、规模化追问（25 条）

31. 你的 RAG 系统现在 50K 文档，如果扩展到 5000 万文档，架构需要怎么改？
32. 如果 QPS 从 10 突然涨到 1000，你的推理服务怎么应对？autoscaling 来得及吗？
33. 单个向量数据库 collection 到 1 亿 vectors 会有什么问题？怎么解决？
34. 如果需要同时服务 100 个不同的模型（不同大小），GPU 资源怎么调度？
35. 你的 embedding 服务如果需要支持 10000 QPS 的 encoding 请求，怎么设计？
36. KV cache 在 128K context 场景下单请求就要几十 GB，怎么在有限 GPU 内存中服务多个这样的请求？
37. 如果你的 LLM serving 需要跨 3 个 region 部署，模型同步和流量路由怎么设计？
38. 100 个租户共享一个 GPU 集群，某个租户突然发大量请求，怎么保护其他租户？
39. 如果 reranker 成为延迟瓶颈（200ms），流量又翻 10 倍，怎么优化？
40. 你的 RAG pipeline 如果需要支持实时数据（<1min 延迟），增量索引怎么设计？
41. 1000 张 GPU 的集群，每周有 2-3 张卡故障，怎么设计自动化故障处理？
42. 如果需要在推理服务中支持 10 种不同的 quantization 配置，怎么管理？
43. 模型从 70B 升级到 405B，serving 架构需要哪些改变？成本增加多少？
44. 如果你的 prefix cache 命中率只有 20%，怎么提升到 80%？
45. 你的系统需要支持 1000 个并发长连接（streaming），网络层怎么设计？
46. 如果 GPU 供给不足（只有申请量的 50%），怎么在有限资源下最大化服务质量？
47. 数据库从单机 Milvus 迁移到分布式集群，数据迁移怎么做到零停机？
48. 如果需要支持多模态（图片+文本）RAG，架构需要哪些改变？
49. 你的推理服务需要从 A100 迁移到 H100，有哪些兼容性问题需要处理？
50. 如果 batch processing 任务量从每天 1M tokens 增长到每天 1B tokens，架构怎么演进？
51. 推理服务需要支持 fine-tuned 模型的热加载（不停服），怎么设计？
52. 如果你的 RAG 系统需要支持 50 种语言，embedding 和检索策略怎么调整？
53. 集群从 8 卡扩展到 1024 卡，NCCL 通信的拓扑优化怎么做？
54. 如果需要在边缘设备（单张 A10G）上服务 70B 模型，有什么方案？
55. 你的系统 P99 延迟要求从 2s 降到 500ms，需要做哪些架构改变？

---

## 三、故障场景追问（25 条）

56. 线上推理服务突然所有请求返回乱码，你怎么排查？
57. GPU utilization 显示 95% 但 throughput 只有预期的 30%，什么原因？
58. 推理服务重启后前 5 分钟延迟特别高，之后恢复正常，为什么？
59. 某个租户反馈间歇性超时，但整体 P99 正常，怎么定位？
60. NCCL AllReduce 突然变慢 10x，但网络监控显示带宽正常，可能是什么原因？
61. 向量数据库查询延迟从 5ms 突然跳到 500ms，怎么排查？
62. 推理服务的 GPU memory 每小时增长 1GB，最终 OOM crash，怎么定位 leak？
63. 模型更新后 A/B test 显示新模型延迟高 50%，但 offline benchmark 没有退化，为什么？
64. 推理服务在凌晨 3 点固定 crash，白天正常，可能是什么原因？
65. 某张 GPU 的推理结果偶尔出错（1/1000 请求），其他卡正常，怎么诊断？
66. Kubernetes pod 频繁被 OOM kill，但 GPU memory 监控显示正常，什么情况？
67. 推理服务的 TPOT 突然从 30ms 跳到 100ms，TTFT 不变，可能原因？
68. RAG 系统的 recall 突然从 92% 降到 70%，没有代码变更，怎么排查？
69. 多卡 TP 推理中，某次请求导致所有 GPU 同时 hang，怎么处理？
70. 推理服务在高负载时出现 "CUDA error: device-side assert triggered"，怎么调试？
71. 客户反馈推理结果不一致（相同输入不同输出），但 temperature=0，为什么？
72. 推理服务的 CPU 使用率突然飙到 100%，GPU 空闲，什么原因？
73. 部署新版本后 prefix cache 命中率从 80% 降到 5%，为什么？
74. 推理服务在 scale up 新节点后，新节点的延迟比旧节点高 3x，为什么？
75. RAG 系统的 reranker 服务突然返回全部相同分数，怎么排查？
76. 推理服务的网络带宽突然打满，但 QPS 没有增加，可能是什么？
77. GPU 温度正常但频率比预期低 20%，nvidia-smi 显示 power limit，怎么处理？
78. 推理服务在特定长度的输入（恰好 2048 tokens）时 crash，其他长度正常，为什么？
79. 分布式推理中某个 rank 的梯度/activation 出现 NaN，怎么定位是哪一层？
80. 推理服务的 health check 通过但实际请求全部超时，什么情况？

---

## 四、设计决策追问（25 条）

81. 为什么选 vLLM 而不是 TensorRT-LLM 或 TGI？各自的优劣势是什么？
82. 为什么用 HNSW 而不是 IVF-PQ？在你的数据规模下 IVF 不是更省内存吗？
83. 为什么选 Milvus 而不是 Pinecone 或 Weaviate？决策依据是什么？
84. 为什么用 cross-encoder rerank 而不是 ColBERT？延迟和精度的 tradeoff 你怎么评估的？
85. 为什么选 BGE 而不是 OpenAI embedding？成本和质量的 tradeoff？
86. 为什么用 chunked prefill 而不是完全的 prefill/decode 分离？
87. 为什么选 EAGLE-3 而不是 Medusa 或 Lookahead decoding？
88. 为什么 KV cache 用 FP16 而不是 FP8？精度和内存的 tradeoff 你测过吗？
89. 为什么用 SSE 而不是 WebSocket 做 streaming？各自的适用场景？
90. 为什么选 Ascend NPU 而不是 NVIDIA GPU？这是技术决策还是业务决策？
91. 为什么 RAG 用 hybrid search 而不是纯 dense？BM25 在你的场景贡献了多少？
92. 为什么 autoscaling 用 queue depth 而不是 GPU utilization 作为指标？
93. 为什么选 RRF 融合而不是 learned fusion？RRF 的 k=60 怎么确定的？
94. 为什么 rerank top-20 而不是 top-50 或 top-10？这个数字怎么确定的？
95. 为什么用 recursive chunking 而不是 semantic chunking？各自的优缺点？
96. 为什么 block size 选 16 而不是 8 或 32？对不同序列长度的影响？
97. 为什么用 LRU 淘汰 prefix cache 而不是 LFU 或 ARC？
98. 为什么选 continuous batching 而不是 dynamic batching with padding？
99. 为什么 TP=8 而不是 TP=4+PP=2？通信模式有什么区别？
100. 为什么用 RAGAS 评估而不是 human evaluation？RAGAS 的局限性是什么？
101. 为什么 RAG pipeline 用同步调用而不是异步 event-driven？
102. 为什么选 Redis 做 exact cache 而不是 Memcached？
103. 为什么用 sentence-aware splitting 而不是固定 token 数切分？
104. 为什么 draft model 用同架构小模型而不是 distilled model？
105. 为什么选 Prometheus+Grafana 而不是 Datadog？成本和功能的 tradeoff？

---

## 五、数据/指标追问（25 条）

106. 你的 EAGLE-3 实现，draft tree 的平均 width 和 depth 分别是多少？
107. 吞吐从 9.22 到 14.30 tok/s，这个是在什么 batch size 下测的？单请求还是并发？
108. TPOT 降低 39%，具体从多少 ms 降到多少 ms？
109. 你的 RAG 系统端到端延迟 P50 和 P99 分别是多少？
110. RAGAS 的四个指标分别是多少？哪个最难提升？
111. 你的向量数据库有多少 vectors？每个 vector 多少维？索引占多少内存？
112. Reranker 处理 20 个 candidates 的延迟具体是多少毫秒？
113. 你的 embedding 服务的 throughput 是多少 queries/second？
114. RAG 系统的 QPS 峰值是多少？日均请求量多少？
115. 你的 prefix cache 命中率在实际生产中是多少？
116. vLLM-Ascend PR 的代码量大概多少行？review 周期多长？
117. Ascend 910B 的 HBM 带宽是多少？和 A100 比差多少？
118. 你的 RAG 系统 empty result rate 是多少？怎么处理 no-result 情况？
119. EAGLE-3 的 draft model 参数量多大？和 target model 的比例是多少？
120. 你的系统可用性是多少个 9？有 SLA 定义吗？
121. GPU memory 中 KV cache 占比多少？模型权重占比多少？
122. 你的 chunking 策略中 overlap 设了多少 tokens？为什么选这个值？
123. Hybrid search 中 BM25 和 dense 的权重比例是多少？怎么调的？
124. 你的 RAG 评估用了多少条 test cases？ground truth 怎么构建的？
125. vLLM-Ascend 的 CI/CD pipeline 跑一次多长时间？覆盖哪些测试？
126. 你的推理服务单卡能支持多少并发请求？
127. EAGLE-3 在不同 sequence length 下的 acceptance rate 变化趋势？
128. 你的 RAG 系统处理一个 query 的 token 消耗（input+output）平均多少？
129. Ascend NPU 上 EAGLE-3 的 kernel 执行时间 breakdown？哪个算子最耗时？
130. 你的项目从开始到 PR 合入花了多长时间？遇到的最大技术障碍是什么？

---

## 六、候选人经验验证追问（25 条）

131. 你的 EAGLE-3 PR 具体改了 vLLM 的哪些模块？能画出调用链吗？
132. PR review 中 reviewer 提了哪些关键意见？你怎么回应的？
133. EAGLE-3 在 Ascend 上实现和在 NVIDIA GPU 上实现有什么本质区别？
134. 你是怎么 debug EAGLE-3 在 Ascend 上的正确性问题的？用了什么工具？
135. vLLM-Ascend 的代码结构你了解多少？scheduler、executor、worker 的关系？
136. 你的 PR 对 vLLM-Ascend 的 CI 有什么影响？有没有引入新的 test case？
137. EAGLE-3 的 feature extraction 层具体是什么结构？参数怎么初始化的？
138. 你在实现过程中参考了 EAGLE-3 原论文的哪些细节？有没有和原实现不同的地方？
139. 吞吐 +55% 这个数字是在什么硬件、什么模型、什么数据集上测的？
140. 你的 RAG 项目是独立负责，团队有几个人？你的具体职责边界是什么？
141. RAG 系统的技术选型是你决定的还是 leader 决定的？你怎么说服团队的？
142. RAGAS 从多少提升到 90% 的？中间做了哪些关键改进？每次提升多少？
143. 你的 RAG 系统有没有遇到过线上故障？怎么处理的？
144. 你说零 CUDA 经验，那 EAGLE-3 在 Ascend 上的算子是怎么实现的？用的什么编程模型？
145. vLLM-MindSpore PR#1020 和 vLLM-Ascend PR#1032 有什么关系？为什么提了两个？
146. 你的 RAG 系统用户量多大？日活多少？有没有做过压测？
147. 你在 vLLM 社区的参与度如何？除了这个 PR 还有其他贡献吗？
148. EAGLE-3 的 draft tree pruning 策略你是怎么实现的？有没有和社区讨论过？
149. 你的 RAG 系统有没有做过 A/B test？怎么证明你的改进是有效的？
150. 如果让你重新做 EAGLE-3 这个 PR，你会有什么不同的做法？
151. 你对 vLLM 的 PagedAttention 实现读过源码吗？能说说关键数据结构？
152. 你的 RAG 系统的 embedding 模型是在线调用还是本地部署？为什么？
153. PR#1032 从提交到合入经历了几轮 review？总共改了多少版？
154. 你在 Ascend NPU 上做性能优化时，profiling 工具用的什么？和 Nsight 有什么区别？
155. 你的 RAG 系统有没有考虑过成本优化？每个 query 的成本大概多少？