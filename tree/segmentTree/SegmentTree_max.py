class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        # nums[ij] - nums[ij-1] >= ij - ij-1
        # nums[i] - i >= nums[j] - j # i>j
        # 定义f[i] 为i结尾的子序列的最大值
        a = [x - i for i, x in enumerate(nums)]
        a = sorted(set(a))
        d = {x: i for i, x in enumerate(a)}
        n = len(nums)
        m = len(a)
        f = [-inf] * n
        st = SegmentTree()
        for i in range(n):
            f[i] = nums[i]
            if i:
                f[i] = max(f[i], nums[i] + st.query(st.root, 0, m - 1, 0, d[nums[i] - i]))
            st.update(st.root, 0, m - 1, d[nums[i] - i], d[nums[i] - i], f[i])
        return max(f)


# 动态开点线段树，区间更新，区间查询（最大值）
class Node:
    __slots__ = ['val', 'left', 'right', 'lazy']

    def __init__(self, left=None, right=None, val=-inf) -> None:
        self.val, self.left, self.right = val, left, right


class SegmentTree:
    def __init__(self):
        self.root = Node()

    # 区间增加val
    def update(self, node, s, e, l, r, val):
        if l <= s and e <= r:
            if val > node.val:
                node.val = val
            return
        mid = (s + e) >> 1
        # 如果是求区间元素和，需要带上区间元素个数
        self.push_down(node, mid - s + 1, e - mid)
        if l <= mid:
            self.update(node.left, s, mid, l, r, val)
        if r > mid:
            self.update(node.right, mid + 1, e, l, r, val)
        self.push_up(node)

    def push_down(self, node, left_num, right_num):
        if not node.left:
            node.left = Node()
        if not node.right:
            node.right = Node()

    def push_up(self, node):
        node.val = max(node.left.val, node.right.val)

    def query(self, node, s, e, l, r):
        if l <= s and e <= r:
            return node.val
        mid = (s + e) >> 1
        self.push_down(node, mid - s + 1, e - mid)
        ans = -inf
        if l <= mid:
            ans = self.query(node.left, s, mid, l, r)
        if r > mid:
            ans = max(ans, self.query(node.right, mid + 1, e, l, r))
        return ans
