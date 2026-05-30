# STAR 周记

## Week 1 (2026-06-01 ~ 2026-06-07)

### Situation
本周开了 row_softmax 主线。

### Task
完成 6 项成熟度。

### Action
按 reference / impl / tests / bench / profile / note 顺序推进。

### Result
- baseline: PyTorch
- final: matches within 1e-5
- improvement: latency 18us -> 12us

### Badcase
非对齐 4097 shape 还差 5%。

### Interview-ready
一句话：在 RTX 4060 上把 softmax 写到 ~412 GB/s。

## Week 2 (待填)
