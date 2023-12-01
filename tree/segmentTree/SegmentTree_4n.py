# lazy 线段树，非动态开点
class Node:
    def __init__(self, l=0, r=0, val=0, lazy=0):
        self.l = l
        self.r = r
        self.val = val
        self.lazy = lazy

class SegmentTree:
    def __init__(self, n):
        self.seg = [Node() for _ in range(n)]

    def push_up(self, o):
        a, b = self.seg[2 * o].val, self.seg[2 * o + 1].val
        self.seg[o].val = a if a >= b else b

    def push_down(self, o):
        v = self.seg[o].lazy
        if v:
            self.seg[o * 2].val += v
            self.seg[o * 2].lazy += v
            self.seg[o * 2 + 1].val += v
            self.seg[o * 2 + 1].lazy += v
            self.seg[o].lazy = 0

    def build_tree(self, o, s, e, f):
        self.seg[o].l = s
        self.seg[o].r = e
        self.seg[o].lazy = 0
        if s == e:
            self.seg[o].val = f[s - 1]
            return
        mid = (s + e) >> 1
        self.build_tree(o * 2, s, mid, f)
        self.build_tree(o * 2 + 1, mid + 1, e, f)
        self.push_up(o)

    def update(self, o, l, r, val):
        if l <= self.seg[o].l and self.seg[o].r <= r:
            self.seg[o].val += val
            self.seg[o].lazy += val
            return
        self.push_down(o)
        mid = (self.seg[o].l + self.seg[o].r) >> 1
        if l <= mid:
            self.update(o * 2, l, r, val)
        if mid < r:
            self.update(o * 2 + 1, l, r, val)
        self.push_up(o)

    def query(self, o, l, r):
        if l <= self.seg[o].l and self.seg[o].r <= r:
            return self.seg[o].val
        self.push_down(o)
        mid = (self.seg[o].l + self.seg[o].r) >> 1
        if r <= mid:
            return self.query(o * 2, l, r)
        if l > mid:
            return self.query(o * 2 + 1, l, r)
        a, b = self.query(o * 2, l, r), self.query(o * 2 + 1, l, r)
        return a if a >= b else b