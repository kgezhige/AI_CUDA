在PyTorch中，`reshape`、`view`和`resize_`是用于调整张量形状的函数，但它们在实现方式和适用场景上有显著区别。以下是它们的详细对比：

---

**1. `view`**
• 功能：调整张量形状，返回与原始张量共享数据的新视图。

• 要求：

  • 张量必须是连续的（contiguous），否则会抛出错误。

  • 新形状的步长（stride）必须与原始存储兼容。

• 特点：

  • 高效，不复制数据。

  • 修改新张量会影响原始张量（共享存储）。

• 示例：

  ```python
  a = torch.arange(4)  # [0, 1, 2, 3]
  b = a.view(2, 2)    # [[0, 1], [2, 3]]
  b[0, 0] = 5         # a也会变为 [5, 1, 2, 3]
  ```

---

**2. `reshape`**
• 功能：调整张量形状，优先返回视图（不复制数据），必要时复制数据以保证连续性。

• 行为：

  • 若原始张量是连续的，行为与`view`相同。

  • 若原始张量不连续，自动调用`contiguous()`生成副本，再调用`view`。

• 特点：

  • 更灵活，但可能引入数据复制（影响性能）。

  • 返回的张量可能与原始张量不共享存储（若复制过数据）。

• 示例：

  ```python
  a = torch.arange(4).view(2, 2).t()  # 转置后不连续
  b = a.reshape(-1)                   # 自动复制数据，得到连续的一维张量
  ```

---

**3. `resize_`**
• 功能：原地修改张量形状，允许调整存储空间大小。

• 行为：

  • 若新尺寸更大：扩展存储空间，新增元素未初始化。

  • 若新尺寸更小：保留原始存储，仅修改形状信息。

• 特点：

  • 是原地操作（带下划线`_`），直接修改原张量。

  • 可能导致数据不一致或未初始化。

• 示例：

  ```python
  a = torch.arange(4)  # [0, 1, 2, 3]
  a.resize_(3, 3)     # 形状变为3x3，新增元素未定义
  ```

---

**关键区别总结**
| 特性                | `view`                  | `reshape`               | `resize_`               |
|---------------------|-------------------------|-------------------------|-------------------------|
| 连续性要求       | 必须连续                | 自动处理非连续          | 无要求                  |
| 共享存储         | 是                     | 可能否（若复制数据）    | 原地修改原张量          |
| 内存分配         | 不复制数据              | 必要时复制数据          | 可能扩展/缩小存储       |
| 新元素初始化     | 不适用                  | 不适用                  | 未初始化                |
| 适用场景         | 高效调整连续张量形状    | 安全处理不确定连续性的张量 | 原地调整尺寸，不关心数据完整性 |

---

**选择建议**
• 使用`view`：当确定张量是连续的，且需要高效共享存储。

• 使用`reshape`：处理不确定是否连续的张量，避免手动检查连续性。

• 使用`resize_`：需要原地修改尺寸，接受潜在的数据未初始化风险。


注意：操作前务必理解数据连续性对计算的影响（如转置、切片等操作会导致非连续张量）。

