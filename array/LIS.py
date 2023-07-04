"""
最长递增子序列LIS
"""
from bisect import bisect_left
from typing import List


def lengthOfLIS(self, nums: List[int]) -> int:
    g = []
    for x in nums:
        i = bisect_left(g, x)
        if i == len(g):
            g.append(x)
        else:
            g[i] = x
    return len(g)


def lengthOfLIS1(self, nums: List[int]) -> int:
    ng = 0
    for x in nums:
        i = bisect_left(nums, x, 0, ng)
        if i == ng:
            ng += 1
        nums[i] = x
    return ng
