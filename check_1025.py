def two_sum(nums,target):
    seen={}
    for idx in range(len(nums)):
        if target-nums[idx] not in seen:
            seen[nums[idx]]=idx
        else:
            return [seen[target-nums[idx]],idx]

print(two_sum([2, 7, 11, 15], 9))   # 期望 [0, 1]
print(two_sum([3, 2, 4], 6))        # 期望 [1, 2]
print(two_sum([3, 3], 6))           # 期望 [0, 1]  ← 重复元素

import numpy as np
scores=np.random.randn(32,128)
row_sum=scores.sum(axis=1,keepdims=True)
assert row_sum.shape==(32,2),row_sum.shape
result =scores/row_sum
assert result.shape==(32,128),result.shape
# import numpy as np
# print(np.arange(24).reshape(2, 3, 4).sum(axis=1).shape)                 # 第 1 组(2,4)
# print(np.arange(24).reshape(2, 3, 4).sum(axis=2, keepdims=True).shape)  # 第 2 组(2,3,1)
# print((np.ones((5, 1)) + np.ones((3,))).shape)                          # 第 3 组(5,3)
# try:                                                                     # 第 4 组：报不报错？报错
#     np.ones((4, 3)) + np.ones((4,))
#     raise RuntimeError("应该报错但没报")
# except ValueError as err:
#     print("第 4 组报错：", err)
# print(np.ones((2, 3, 4)).transpose(2, 0, 1).shape)                      # 第 5 组（4，2，3）