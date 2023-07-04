# 数组线段树 单点更新，区间查询（求和）
class Node:
    __slots__ = ['val', 'left', 'right']

    def __init__(self, left=None, right=None, val=0) -> None:
        self.left, self.right, self.val = left, right, val


class SegmentTree:
    def __init__(self, nums):
        n = len(nums)
        self.nums = nums
        self.root = self.build_tree(nums, 0, n - 1)

    def build_tree(self, nums, s, e):
        if s > e:
            return None
        node = Node(s,e)
        if s == e:
            node.val = nums[s]
        else:
            mid = (s + e) >> 1
            node.left = self.build_tree(nums, s, e)
            node.right = self.build_tree(nums,mid + 1, e)
            node.val = node.left.val + node.right.val
        return node

    # 区间增加val
    def update(self, node, s, e, index, val):
        if s == e:
            node.val = val
            return node
        mid = (s + e) >> 1
        if index <= mid:
            node.left = self.update(node.left, s, mid, index, val)
        if index > mid:
            node.right = self.update(node.right, mid + 1, e, index, val)
        node.val = node.left.val + node.right.val
        return node

    def query(self, node, s, e, l, r):
        if l <= s and e <= r:
            return node.val
        mid = (s + e) >> 1
        ans = 0
        if l <= mid:
            ans += self.query(node.left, s, mid, l, r)
        if r > mid:
            ans += self.query(node.right, mid + 1, e, l, r)
        return ans
