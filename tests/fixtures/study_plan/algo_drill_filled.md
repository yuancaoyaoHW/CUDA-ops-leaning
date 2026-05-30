# 算法 / C++ Drill

## Week 1 (2026-06-01)

### Algo: TopK
- 出处: leetcode 215
- 思路: 用大小为 k 的最小堆维护
- 复杂度: O(n log k)
- 关键代码片段:
  ```cpp
  priority_queue<int, vector<int>, greater<int>> pq;
  ```
- 卡点: heap 初始化用 vector 还是逐个 push 性能差异

### Cpp: RAII
- 主题: RAII 与 unique_ptr
- 学到: 析构函数顺序与异常安全
- 一段最小代码示例:
  ```cpp
  unique_ptr<int> p = make_unique<int>(42);
  ```
- 自测题: 为什么 RAII 比 try/finally 更安全

## Week 2 (待填)
