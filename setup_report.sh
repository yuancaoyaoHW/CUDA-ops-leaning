set -euo pipefail

WORKDIR="$HOME/claude-token-burn/Awesome-LLM-Inference"
mkdir -p "$(dirname "$WORKDIR")"

if [ ! -d "$WORKDIR/.git" ]; then
	  git clone https://github.com/yuancaoyaoHW/Awesome-LLM-Inference.git "$WORKDIR"
fi

cd "$WORKDIR"
mkdir -p reports

claude auth status --text || true
claude --version

CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude \
	  --model opus \
	    --effort max \
	      --teammate-mode tmux \
	          --dangerously-skip-permissions \
		    --name "awesome-llm-inference-deep-paper-team" \
		      "$(cat <<'PROMPT'
你现在要用 Agent Team 对当前仓库进行极高强度、极高 token 消耗、极细粒度的论文与技术体系分析。目标仓库是：
https://github.com/yuancaoyaoHW/Awesome-LLM-Inference

总目标：
把这个仓库中的 LLM inference 论文、项目、系统、benchmark、技术路线，分析成一套可用于汇报、学习、复现、研究选题和工程落地的完整知识体系。不要只做普通 README 总结。要进行多轮拆解、交叉验证、争论、重新分类、知识图谱化、工程路线化。

工作边界：
1. 可以读取、搜索、分析仓库内所有文件。
2. 可以写入 reports/ 目录。
3. 不要修改仓库原始文件，除非它们位于 reports/。
4. 能联网时，尽量根据 README/Markdown 中的论文链接下载或读取论文摘要、arXiv 页面、PDF 文本；不能联网或解析失败时，要记录失败原因。
5. 需要把所有中间发现沉淀到 reports/，不能只在对话里输出。
6. 不要过早结束。至少完成 4 轮：扫描、分类、逐篇深读、交叉辩论与综合。

请创建一个 Agent Team。需要 16 个 teammate，角色如下：

A. lead-coordinator：
- 负责拆任务、维护共享任务列表、综合所有结论。
- 要求每个 teammate 产出结构化报告。
- 组织至少 2 轮交叉批判。

B. repo-cartographer：
- 全仓库扫描。
- 识别 README、论文列表、分类结构、链接、项目、benchmark、代码片段。
- 输出 reports/01_repo_map.md 和 reports/01_repo_inventory.json。

C. paper-extractor：
- 抽取所有论文条目。
- 生成标准字段：title, authors, year, venue, url, arxiv_id, category, keywords, claimed_problem, claimed_method, claimed_result。
- 输出 reports/02_paper_catalog.csv 和 reports/02_paper_catalog.json。

D. taxonomy-architect：
- 建立 LLM inference 技术分类体系。
- 分类至少覆盖：prefill、decode、attention kernel、KV cache、PagedAttention、continuous batching、speculative decoding、prefix caching、quantization、MoE inference、distributed serving、parallelism、scheduling、memory management、benchmark、hardware backend、NPU/GPU/TPU、edge inference、serving framework。
- 输出 reports/03_taxonomy.md。

E. systems-historian：
- 按时间线梳理 LLM inference 系统演进。
- 比较 vLLM、SGLang、TensorRT-LLM、llama.cpp、DeepSpeed-FastGen、LightLLM、TGI、LMDeploy、MLC-LLM 等系统的技术定位。
- 输出 reports/04_systems_lineage.md。

F. kernel-mathematician：
- 重点分析 attention kernel、FlashAttention、PagedAttention、GQA/MQA、KV cache layout、memory bandwidth、roofline、batching 对性能的影响。
- 给出数学推导、复杂度、显存占用公式、吞吐瓶颈。
- 输出 reports/05_kernel_and_math.md。

G. serving-scheduler：
- 分析 serving 层：request scheduling、continuous batching、iteration-level scheduling、chunked prefill、prefix reuse、SLO-aware scheduling、disaggregated prefill/decode。
- 输出 reports/06_serving_scheduling.md。

H. kv-cache-specialist：
- 专门分析 KV cache 管理：分页、压缩、量化、淘汰、共享、前缀缓存、多租户隔离。
- 输出 reports/07_kv_cache.md。

I. speculative-decoding-specialist：
- 专门分析 speculative decoding、Medusa、EAGLE、Lookahead、draft model、tree decoding、multi-token prediction。
- 对比适用场景、理论加速上限、失败条件。
- 输出 reports/08_speculative_decoding.md。

J. quantization-and-compression：
- 分析 INT8/INT4/FP8/AWQ/GPTQ/SmoothQuant/KV quantization/sparsity。
- 区分 weight-only、activation-aware、KV cache quant、serving-time quant。
- 输出 reports/09_quantization.md。

K. distributed-inference-engineer：
- 分析 tensor parallelism、pipeline parallelism、expert parallelism、sequence parallelism、data parallel serving、multi-node serving、disaggregated architecture。
- 输出 reports/10_distributed_inference.md。

L. benchmark-evaluator：
- 梳理 benchmark 和指标：TTFT、TPOT、latency、throughput、QPS、tokens/s、GPU utilization、memory fragmentation、SLO violation、cost/token。
- 设计一套 benchmark matrix。
- 输出 reports/11_benchmark_map.md。

M. implementation-planner：
- 把论文方法映射到工程实现。
- 生成“从论文到代码”的复现路线，包括环境、数据、关键模块、实验指标、风险。
- 输出 reports/12_reproduction_plan.md。

N. contradiction-hunter：
- 专门找冲突、重复、分类错误、宣传性结论、证据不足、实验不公平、benchmark 不可比之处。
- 对所有 teammate 的报告做批判。
- 输出 reports/13_contradictions_and_caveats.md。

O. knowledge-graph-builder：
- 构建论文-技术-系统-指标-硬件之间的知识图谱。
- 输出 reports/14_knowledge_graph.md、reports/14_knowledge_graph.json、reports/14_mermaid_graphs.md。

P. report-and-slide-writer：
- 生成总报告、汇报大纲和可讲解版本。
- 输出 reports/00_MASTER_REPORT.md、reports/15_presentation_outline.md、reports/16_reading_plan.md、reports/17_research_ideas.md。

执行要求：
1. 先由 repo-cartographer 和 paper-extractor 扫描仓库。
2. lead-coordinator 根据扫描结果把论文分给各 teammate。
3. 每个 teammate 至少产出：
   - 任务范围
   - 读取材料
   - 技术分类
      - 核心方法
         - 数学/系统机制
	    - 工程影响
	       - 局限
	          - 需要复核的问题
		  4. contradiction-hunter 必须阅读其他 teammate 的报告，并写出批判意见。
		  5. lead-coordinator 必须组织一次“技术路线辩论”：
   - vLLM/PagedAttention 路线
   - SGLang/RadixAttention 路线
   - TensorRT-LLM/编译优化路线
   - llama.cpp/端侧路线
      - speculative decoding 路线
         - disaggregated serving 路线
	 6. 需要输出一个“LLM inference 技术总地图”，用 Markdown + Mermaid 表示。
7. 需要输出一个“读论文顺序”，分为：
   - 工程入门
   - 系统架构
   - kernel/算子
   - serving/scheduling
      - memory/KV cache
         - speculative decoding
	    - quantization
	       - distributed inference
	          - benchmark/evaluation
		  8. 需要输出一个“未来研究选题库”，至少 50 条，每条包含：
		     - problem
		        - motivation
			   - possible method
			      - required papers
			         - expected experiment
				    - engineering difficulty
				    9. 所有报告用中文写，保留关键英文术语。
				    10. 对“革命性、颠覆性、范式转移”等宣传性措辞直接忽略，只分析方法、证据、结果。
11. 所有结论都要尽量绑定到具体论文、系统或 repo 条目。
12. 完成后给出 reports/ 下所有文件清单和每个文件用途。

追加高消耗要求：
1. 每个 teammate 不允许只给摘要，必须做逐条论文级别分析。
2. 每篇论文至少生成：
   - 100-300 字中文摘要
   - 方法机制
   - 关键假设
   - 与同类方法差异
   - 实验指标
      - 可复现难点
   - 对 vLLM/SGLang/TensorRT-LLM/llama.cpp 的潜在影响
3. 每个 teammate 完成初稿后，必须把自己的结论发给 contradiction-hunter。
4. contradiction-hunter 至少提出 100 条具体批判意见。
5. lead-coordinator 必须要求每个 teammate 对批判意见逐条回应。
6. lead-coordinator 在所有回应后，生成 consensus、disagreement、open questions 三个部分。
7. 每个核心技术方向至少画 1 个 Mermaid 图。
8. 对每个方向给出“面向 A30/V100/4090/Ascend NPU 的工程学习路线”。
9. 输出 reports/18_token_heavy_appendix.md，收录所有长表格、交叉对比、争论记录、失败记录、未读论文清单。
PROMPT
)"
