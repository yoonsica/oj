"""
并查集 非递归版本
"""
from collections import defaultdict


class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}
        self.rank = defaultdict(int)

    def find(self, x):
        """
        查找根节点
        路径压缩
        """
        if self.father[x]:
            self.father[x] = self.find(self.father[x])
        return self.father[x]

    def union(self, x, y):
        """
        合并两个节点
        """
        x, y = self.find(x), self.find(y)
        if x != y:
            if self.rank[x] < self.rank[y]:
                self.father[x] = y
            else:
                self.father[y] = x
                if self.rank[x] == self.rank[y]:
                    self.rank[x] += 1

    def is_connected(self, x, y):
        """
        判断两节点是否相连
        """
        return self.find(x) == self.find(y)

    def add(self, x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None
