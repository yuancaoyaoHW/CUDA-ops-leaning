# 投递邮件模板

> 不得虚构经历，所有内容基于已验证事实
> 三种冷邮件风格 + 内推请求模板

---

## 一、冷邮件模板

### 风格 A：技术导向（适合技术团队 leader）

**Subject**: vLLM EAGLE-3 贡献者 — 对贵团队推理优化方向感兴趣

您好，

我是袁曹尧，浙江大学计算机技术硕士（2025 年毕业）。我在 vLLM-Ascend 社区独立实现了 EAGLE-3 推测解码（PR #1032 已合入主线），在昇腾 NPU 上实现了 55% 的吞吐提升。

我注意到贵团队在 [具体方向] 方面的工作，这与我的经验方向高度相关。我的核心能力：

- 推测解码全链路实现：draft model → KV cache → rejection sampler，有量化性能数据
- 开源协作：通过社区 code review 合入 PR 的完整经验
- RAG 系统独立交付：RAGAS 评测准确率 90%

我的 GitHub：github.com/yuancaoyaoHW
PR 链接：[vLLM-Ascend #1032]

如果方便，希望能有 15 分钟的时间了解贵团队的技术方向和岗位需求。

袁曹尧

---

### 风格 B：简洁直接（适合 HR / 招聘负责人）

**Subject**: 大模型推理系统工程师 — 浙大硕士 / vLLM 社区贡献者

您好，

我是袁曹尧，正在寻找大模型推理系统方向的工作机会。简要背景：

- 学历：浙江大学计算机技术硕士
- 核心项目：独立实现 EAGLE-3 推测解码，PR 合入 vLLM-Ascend 社区主线
- 量化成果：吞吐 +55%，延迟 -39%（Atlas 310P3）
- 其他：RAG 系统独立交付，RAGAS 评测 90% 准确率

我对贵公司的 [岗位名称] 很感兴趣。附上简历供参考，期待有机会进一步交流。

袁曹尧
电话：153-3248-1217
GitHub：github.com/yuancaoyaoHW

---

### 风格 C：故事导向（适合创业公司 / 技术氛围浓的团队）

**Subject**: 从 NPU 推测解码到 vLLM 社区 PR 合入 — 寻找推理优化方向机会

您好，

我花了几个月时间在昇腾 NPU 上从零实现了 EAGLE-3 推测解码。这个过程中我需要理解 vLLM V1 的 scheduler、model runner、KV cache manager 如何协作，然后设计适配方案打通完整的 draft-verify pipeline。最终 PR 合入了 vLLM-Ascend 社区主线，吞吐提升了 55%。

这段经历让我对推理系统的核心问题有了实战理解：KV cache 如何管理、推测解码的 acceptance rate 如何调优、如何设计 benchmark 验证性能。

我同时有 RAG 系统的独立交付经验——从文档解析到答案生成的完整链路，RAGAS 评测 90% 准确率。

我的短板是目前没有 CUDA kernel 开发经验，正在系统学习中。如果贵团队看重推理系统理解和开源协作能力，愿意给一个补强 CUDA 的成长空间，我很希望能聊聊。

袁曹尧
GitHub：github.com/yuancaoyaoHW

---

## 二、内推请求模板

### 模板 1：认识的人（同学/前同事）

**Subject**: 想请你帮忙内推 [公司名] 的 [岗位名]

[称呼] 你好，

好久没联系了，希望一切顺利。

我最近在找大模型推理系统方向的工作，看到 [公司名] 有一个 [岗位名] 的 opening，想请你帮忙内推一下。

简单说下我的背景：
- 浙大计算机硕士，2025 年毕业
- 在 vLLM-Ascend 社区实现了 EAGLE-3 推测解码，PR #1032 已合入主线
- 在昇腾 NPU 上实现了 55% 吞吐提升
- 还独立做过一个 RAG 系统后端，准确率 90%

我觉得这个岗位和我的经验比较匹配，特别是 [具体匹配点]。

如果方便的话，我把简历发给你，麻烦帮忙推一下。如果你觉得不太合适也没关系，直接告诉我就好。

谢谢！

袁曹尧

---

### 模板 2：弱关系（社区认识 / LinkedIn 联系人）

**Subject**: 请教 [公司名] 推理团队的情况 + 内推请求

[称呼] 你好，

我是袁曹尧，之前在 [场景：vLLM 社区 / 技术会议 / LinkedIn] 有过交流。

我目前在找大模型推理系统方向的工作，注意到你在 [公司名] 的 [团队名]。想请教两个问题：

1. 你们团队目前在推理优化方面主要关注哪些方向？
2. 是否有适合我背景的 opening？

我的核心经验是在 vLLM 生态中实现 EAGLE-3 推测解码（PR #1032 合入 vLLM-Ascend），在 NPU 上实现了 55% 吞吐提升。也有 RAG 系统独立交付经验。

如果你觉得我的背景可能匹配，希望能帮忙内推或者介绍一下团队情况。如果不太合适，也欢迎给些建议。

谢谢你的时间！

袁曹尧
GitHub：github.com/yuancaoyaoHW

---

## 三、使用注意事项

1. **[方括号] 内容必须替换**：公司名、岗位名、具体匹配点等需要根据实际情况填写
2. **不要群发相同内容**：每封邮件至少定制 1-2 句与目标公司/团队相关的内容
3. **不要虚构**：所有技术声明均基于已验证事实，不可添加未完成的项目或虚构的数据
4. **附上 PR 链接**：技术岗位的邮件中附上 PR 链接比附简历更有说服力
5. **控制长度**：冷邮件不超过 200 字（正文），内推请求不超过 150 字
6. **跟进策略**：发送后 5-7 个工作日无回复，可发一次简短跟进（不超过 3 句话）

---

## 四、英文版冷邮件模板

### Technical Style (for engineering managers)

**Subject**: vLLM EAGLE-3 Contributor — Interested in Your Inference Optimization Team

Hi [Name],

I'm Yuan Caoyao, a recent CS Master's graduate from Zhejiang University. I independently implemented EAGLE-3 speculative decoding for vLLM-Ascend (PR #1032, merged), achieving 55% throughput improvement on Ascend NPU.

I noticed your team's work on [specific area]. My relevant experience:

- Full speculative decoding pipeline: draft model → KV cache → rejection sampler, with quantified performance gains
- Open-source collaboration: PR merged through community code review
- RAG system delivery: 90% accuracy on RAGAS evaluation

GitHub: github.com/yuancaoyaoHW

Would you have 15 minutes to discuss your team's direction and potential fit?

Best,
Yuan Caoyao

---

### Concise Style (for recruiters)

**Subject**: LLM Inference Engineer — Zhejiang University MS / vLLM Contributor

Hi,

I'm looking for LLM inference system roles. Quick background:

- MS in Computer Science, Zhejiang University (2025)
- Implemented EAGLE-3 speculative decoding, PR merged into vLLM-Ascend
- Results: 55% throughput improvement, 39% latency reduction
- Also: independent RAG system delivery, 90% accuracy

I'm interested in [role name] at [company]. Resume attached — happy to discuss further.

Best,
Yuan Caoyao
GitHub: github.com/yuancaoyaoHW
