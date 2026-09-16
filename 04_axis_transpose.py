# 第 10 组：(32,128) / (32,)
#   纸上：报错 ｜ 实际：报错 ✅
#   ValueError: operands could not be broadcast together with shapes (32,128) (32,)
#   ⇒ 报错自己把两个 shape 打出来了，右端对齐逐列查：128 vs 32
#   ⚠️ 今天运气好在 32 ≠ 128 所以报错了。
#      如果 batch 恰好等于 hidden dim（比如 (128,128) / (128,)），
#      它会静默跑通、逐列除、数值全错、零报错 ← 4.7 存在的理由
import numpy as np
print(np.arange(24).reshape(4,6).sum(axis=0).shape)              # 期望 (6,)
print(np.arange(24).reshape(4,6).sum(axis=1).shape)              # 期望 (4,)
print(np.arange(24).reshape(4,6).sum(axis=1, keepdims=True).shape)  # 期望 (4,1) ← 纸上写的 (4,6)，错
print(np.arange(24).reshape(4,6).sum().shape)                    # 期望 ()
print(np.random.randn(2,3,4).sum(axis=1).shape)                  # 期望 (2,4)
print(np.random.randn(2,3,4).transpose(1,0,2).shape)             # 期望 (3,2,4)
print(np.random.randn(32,128).T.shape)                           # 期望 (128,32)
print(np.random.randn(2,3,4).T.shape)                            # 期望 (4,3,2)
x = np.random.randn(32, 128)
try:
    x / x.sum(axis=1)
    raise RuntimeError("应该报错但没报")     # ← 走到这一行说明出事了
except ValueError as e:
    print("第 10 组按预期报错：", e)
row_sum = x.sum(axis=1, keepdims=True)
assert row_sum.shape == (32, 1), row_sum.shape

probs = x / row_sum
assert probs.shape == (32, 128), probs.shape
print("第 11 组 keepdims 广播成功：", probs.shape)