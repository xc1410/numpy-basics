# 向量化 vs for 循环：一百万个元素求平方和
# 三次倍数：___、___、___
# 坑：第一版漏了 .sum()，只做了平方没求和，量出 53.1 倍是虚高的。
#     报错在最后那行数值 assert 才暴露：ValueError: truth value of an array is ambiguous
#     ⇒ 一个数组被当成条件判真假，说明它本该是标量却不是
# 向量化 vs for 循环：一百万个元素求平方和
# 三次倍数：56.6 / 62.7 / 65.2，波动约 ±7%（自己机器；LeetCode 判题机是 ±35%）
# for 循环侧 0.1606/0.1605 几乎不动，波动全在向量化那 2.5 ms 上
#   ⇒ 测量越短的东西，相对误差越大
# 六节 向量化计时：一百万元素求平方和
#   for 循环 0.1606 s ／ 向量化 0.0026 s
#   三次倍数：56.6 / 62.7 / 65.2  ⇒ 报"约 60 倍"，本机波动 ±7%
#   for 循环侧 0.1606 / 0.1605 几乎不动，波动全在向量化那 2.5 ms 上
#   ⇒ 测量越短的东西，相对误差越大
#
# BUG 记录（4.1 第 4 条样本）：第一版向量化漏写 .sum()
#   零报错，打印"快了 53.1 倍"——只平方没求和，倍数虚高
#   抓住它的是最后那行【差点没写】的数值 assert：
#     ValueError: The truth value of an array with more than one element is ambiguous
#   这个报错的含义是"它本该是标量却不是"，九成是漏了 .sum() / .mean() 这类聚合
#   ⇒ 53.1 比真值 60 更好看、更容易让人相信。最坏的 bug 是给出一个你愿意相信的答案的那种
import numpy as np
matrix = np.arange(12).reshape(3,4)
assert matrix.shape == ( 3, 4),matrix.shape
row_0 = matrix[0]
assert row_0.shape == (4,),row_0.shape
col_0 = matrix[:,0]
assert col_0.shape == (3,),col_0.shape
single = matrix[1,2]
print(type(single),single)
block = matrix[1:,1:3]
assert block.shape == (2,2),block.shape

#2
flat = np.arange(12)
assert flat.shape == (12,),flat.shape
a = flat.reshape(3,4)
assert a.shape == (  3 , 4  ), a.shape
b = flat.reshape(2, 6)
assert b.shape == ( 2  , 6  ), b.shape
c = flat.reshape(3, -1)
assert c.shape == (   3,   4), c.shape
d = flat.reshape(-1, 2)
assert d.shape == (  6 ,  2 ), d.shape
e = a.reshape(-1)
assert e.shape == ( 12 ,), e.shape
t = a.T
assert t.shape == (  4 ,3  ), t.shape
# 12 个元素塞不进 3×3，这行会报错。跑一次，看报错怎么说，然后注释掉
# bad = flat.reshape(3, 3)
#ValueError: cannot reshape array of size 12 into shape (3,3)
import time
data = np.random.randn(1000000)
assert data.shape == (1000000,),data.shape

start = time.perf_counter()
total_loop = 0.0
for value in data:
    total_loop += value**2
loop_time = time.perf_counter() -start

start = time.perf_counter()
total_vec = (data **2).sum()
vec_time = time.perf_counter() -start

print(f"for 循环 {loop_time:.4f} 秒")
print(f"向量化   {vec_time:.4f} 秒")
print(f"快了 {loop_time / vec_time:.1f} 倍")

assert abs(total_loop - total_vec)<1e-6,(total_loop,total_vec)
# 4 广播
# 规则：两个 shape 右端对齐，左边不够的前面补 1，然后从右往左逐维比较
#   相等 → 用它 ｜ 有一个是 1 → 拉伸 ｜ 都不是 → 报错
#
# 组 1 (3,4)+(4,)  → (3,4)   4 对 4，3 对补出来的 1
# 组 3 (3,4)+(3,)  → 报错    (3,) 的 3 被对到最后一维去和 4 比，不是和第一维的 3 比
#      ⚠️ 我在这组判断错了：广播不找"哪里有相同的数字"，只从右往左硬对
# 组 4 (32,1)+(1,32) → (32,32)  两边各拉伸一次，不报错
#      ⚠️ 本想要 32 个数，拿到 1024 个。程序照跑、loss 照降、零报错
#      ⇒ 这就是 4.7 强制写 assert shape 的理由
# 4 广播
# 组 1
zeros_3x4 = np.zeros((3, 4))
ones_4 = np.ones((4,))
r1 = zeros_3x4 + ones_4
assert r1.shape == ( 3  ,4   ), r1.shape

# 组 2：纸上先写，会报错吗？
col_3x1 = np.zeros((3, 1))
row_1x4 = np.ones((1, 4))
r2 = col_3x1 + row_1x4
assert r2.shape == (   3, 4  ), r2.shape

# 组 3：纸上先写，会报错吗？先注释着 
# r3 = np.zeros((3, 4)) + np.ones((3,))

# 组 4：不报错，但这是深度学习里最贵的一类 bug
silent = np.zeros((32, 1)) + np.ones((1, 32))
assert silent.shape == (   32,  32 ), silent.shape