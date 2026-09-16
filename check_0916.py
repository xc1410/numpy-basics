def longest_consecutive(nums):
    longest_nums=1
    length=1
    seen=sorted(set(nums))
    if len(seen)<=1:
        return len(seen)
    for i in range(1,len(seen)):
        if seen[i]-seen[i-1]==1:
            length+=1
            longest_nums=max(longest_nums,length)
        else:
            length=1
    return longest_nums
print(longest_consecutive([100, 4, 200, 1, 3, 2]))              # 期望 4
print(longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))      # 期望 9   ← 含重复的 0，证伪数据
print(longest_consecutive([]))                                   # 期望 0   ← 边界，longest 初值写 1 就挂
print(longest_consecutive([1, 0, 1, 2]))                         # 期望 3   ← 重复 + 不连续起点
