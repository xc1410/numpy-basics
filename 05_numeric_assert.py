import numpy as np
print(0.1+0.2)
print(np.allclose(np.array([0.1+0.2,1.0]),np.array([0.3,1.0])))
print(np.allclose(np.array([1.0,1.0]),1))

scores = np.arange(1, 10).reshape(3, 3)
print(scores)                    # 期待三行：[[1 2 3] / [4 5 6] / [7 8 9]]
col_sum = scores.sum(axis=0)
assert col_sum.shape == (3,), col_sum.shape
print(col_sum)                   # 期待 [12 15 18]
col_probs = scores / col_sum
assert col_probs.shape == (3, 3), col_probs.shape
print(col_probs.sum(axis=0))     # 期待 [1. 1. 1.]
assert np.allclose(col_probs.sum(axis=0), 1), col_probs.sum(axis=0)

row_sum_flat = scores.sum(axis=1)
print(row_sum_flat)              # 期待 [ 6 15 24]
wrong = scores / row_sum_flat
assert wrong.shape == (3, 3), wrong.shape       # ← shape assert 通过了
print(wrong.sum(axis=1))         # 期待 [0.425 1.25  2.075]
try:
    assert np.allclose(wrong.sum(axis=1), 1), wrong.sum(axis=1)
    raise RuntimeError("数值 assert 应该挂但没挂")
except AssertionError as err:
    print("数值 assert 按预期挂了：", err)      # 期待 数值 assert 按预期挂了： [0.425 1.25  2.075]

row_sum = scores.sum(axis=1, keepdims=True)
assert row_sum.shape == (3, 1), row_sum.shape
row_probs = scores / row_sum
assert row_probs.shape == (3, 3), row_probs.shape
assert np.allclose(row_probs.sum(axis=1), 1), row_probs.sum(axis=1)
print(row_probs)   # 期待三行，第 0 行 [0.16666667 0.33333333 0.5       ]，第 2 行 [0.29166667 0.33333333 0.375     ]

elementwise = scores * scores
matmul = scores @ scores
assert elementwise.shape == (3, 3), elementwise.shape
assert matmul.shape == (3, 3), matmul.shape     # 两个 shape assert 都过，分不出来
print(elementwise)               # 期待三行：[[ 1  4  9] / [16 25 36] / [49 64 81]]
print(matmul)                    # 期待三行：[[ 30  36  42] / [ 66  81  96] / [102 126 150]]

mask = scores > 4
print(mask)          # 三行：[[False False False] / [False  True  True] / [ True  True  True]]
print(scores[mask])  # [5 6 7 8 9]
print(mask.sum())    # 5

picked = scores[mask]
assert picked.ndim == 1, picked.ndim
assert picked.shape == (mask.sum(),), picked.shape
assert (picked > 4).sum() == picked.shape[0], picked    # 挑出来的每个都 > 4

grid = np.arange(6).reshape(2, 3)
print((grid > 2).shape)                                        # 第 1 组：shape(2,3)
print(grid[grid > 2])                                          # 第 2 组：具体值[3,4,5,]
print((grid > 2).sum())                                        # 第 3 组：具体值3
try:                                                           # 第 4 组：报不报错？报错
    np.ones((2, 3)) / np.ones((2, 3)).sum(axis=1)
    raise RuntimeError("应该报错但没报")
except ValueError as err:
    print("第 4 组报错：", err)
print((np.ones((3, 3)) / np.ones((3, 3)).sum(axis=1)).shape)   # 第 5 组：shape（3，3）