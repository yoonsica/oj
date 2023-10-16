# 动态开点线段树 单点更新，区间查询（求和）
class Node:
    __slots__ = ['val', 'left', 'right']

    def __init__(self, left=None, right=None, val=0) -> None:
        self.left, self.right, self.val = left, right, val


class SegmentTree:
    def __init__(self):
        self.root = Node()

    # 单点增加val
    def update_index(self, node, s, e, index, val):
        if s == e:
            node.val = val
            return node
        self.push_down(node)
        mid = (s + e) >> 1
        if index <= mid:
            self.update_index(node.left, s, mid, index, val)
        if index > mid:
            self.update_index(node.right, mid + 1, e, index, val)
        self.push_up(node)

    # 区间更新
    def update(self, node, s, e, l, r, val):
        if l <= s and e <= r:
            node.val = val
            return node
        mid = (s + e) >> 1
        if l <= mid:
            self.update(node.left, s, mid, l, r, val)
        if r > mid:
            self.update(node.right, mid + 1, e, l, r, val)
        self.push_up(node)


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

    def push_up(self,node):
        node.val = node.left.val + node.right.val

    def push_down(self, node):
        if not node.left:
            node.left = Node()
        if not node.right:
            node.right = Node()