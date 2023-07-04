# 动态开点线段树 单点更新，区间查询（求和）
class Node:
    __slots__ = ['val', 'left', 'right']

    def __init__(self, left=None, right=None, val=0) -> None:
        self.left, self.right, self.val = left, right, val


class SegmentTree:
    def __init__(self):
        self.root = Node()

    # 单点增加val
    def update(self, node, s, e, index, val):
        # 动态开点
        if not node:
            node = Node()
        if s == e:
            node.val = val
            return node
        mid = (s + e) >> 1
        if index <= mid:
            node.left = self.update(node.left, s, mid, index, val)
        if index > mid:
            node.right = self.update(node.right, mid + 1, e, index, val)
        node.val = (node.left.val if node.left else 0) + (node.right.val if node.right else 0)
        return node

    # 也可以这样写
    # def update(self, node, s, e, index, val):
    #     # 动态开点
    #     if not node.left:
    #         node.left = Node()
    #     if not node.right:
    #         node.right = Node()
    #     if s == e:
    #         node.val = val
    #         return
    #     mid = (s + e) >> 1
    #     if index <= mid:
    #         self.update(node.left, s, mid, index, val)
    #     if index > mid:
    #         self.update(node.right, mid + 1, e, index, val)
    #     node.val = node.left.val + node.right.val

    def query(self, node, s, e, l, r):
        if not node:
            return 0
        if l <= s and e <= r:
            return node.val
        mid = (s + e) >> 1
        ans = 0
        if l <= mid:
            ans += self.query(node.left, s, mid, l, r)
        if r > mid:
            ans += self.query(node.right, mid + 1, e, l, r)
        return ans
