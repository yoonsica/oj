# 动态开点线段树，区间更新(加法），区间查询（求和）
class Node:
    __slots__ = ['val', 'left', 'right', 'lazy']

    def __init__(self, left=None, right=None, val=0, lazy=0) -> None:
        self.val, self.left, self.right, self.lazy = val, left, right, lazy


class SegmentTree:
    def __init__(self):
        self.root = Node()

    # 区间增加val
    def update(self, node, s, e, l, r, val):
        if l <= s and e <= r:
            node.val += (e - s + 1) * val
            node.lazy += val
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
        if node.lazy == 0:
            return
        node.left.val += node.lazy * left_num
        node.right.val += node.lazy * right_num
        node.left.lazy += node.lazy
        node.right.lazy += node.lazy
        node.lazy = 0

    def push_up(self, node):
        node.val = node.left.val + node.right.val

    def query(self, node, s, e, l, r):
        if l <= s and e <= r:
            return node.val
        mid = (s + e) >> 1
        self.push_down(node, mid - s + 1, e - mid)
        ans = 0
        if l <= mid:
            ans += self.query(node.left, s, mid, l, r)
        if r > mid:
            ans += self.query(node.right, mid + 1, e, l, r)
        return ans


