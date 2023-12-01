class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # 枚举[(1 << 4) - 1,1 << 20)区间内，大小为k的子集，
        s = (1 << k) - 1
        ans = []
        while s < (1 << n):
            ans.append([i + 1 for i in range(n + 1) if s >> i & 1])
            lb = s & -s
            left = s + lb
            right = ((s ^ (s + lb)) // lb) >> 2
            s = left | right
        return ans
