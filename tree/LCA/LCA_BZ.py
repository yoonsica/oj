from collections import deque
from math import inf
from typing import List
# 倍增法求LCA
class TreeAncestor:
    def __init__(self,root, edges: List[List[int]]):
        n = len(edges) + 1
        m = n.bit_length()
        g = [[] for _ in range(n)]
        for x, y in edges:  # 节点编号从 0 开始
            g[x].append(y)
            g[y].append(x)

        depth = [inf] * n
        depth[root] = 0
        pa = [[-1] * m for _ in range(n)]

        def bfs(root):
            q = deque([root])
            while q:
                u = q.popleft()
                for v in g[u]:
                    if depth[v] > depth[u] + 1:
                        depth[v] = depth[u] + 1
                        q.append(v)
                        pa[v][0] = u
                        for k in range(m - 1):
                            pa[v][k + 1] = pa[pa[v][k]][k]

        bfs(root)

        self.depth = depth
        self.pa = pa

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(k.bit_length()):
            if (k >> i) & 1:  # k 二进制从低到高第 i 位是 1
                node = self.pa[node][i]
        return node

    # 返回 x 和 y 的最近公共祖先（节点编号从 0 开始）
    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        # 使 y 和 x 在同一深度
        y = self.get_kth_ancestor(y, self.depth[y] - self.depth[x])
        if y == x:
            return x
        for i in range(len(self.pa[x]) - 1, -1, -1):
            px, py = self.pa[x][i], self.pa[y][i]
            if px != py:
                x, y = px, py  # 同时上跳 2**i 步
        return self.pa[x][0]


# 改良版本
from typing import List
from collections import defaultdict, deque
import sys
from math import inf
m = 16 # 按需修改
class TreeAncestor:
    def __init__(self, n, edges: List[List[int]]):
        g = [[] for _ in range(n)]
        for x, y in edges:  # 节点编号从 0 开始
            g[x].append(y)
            g[y].append(x)

        depth = [inf] * n
        depth[root] = 0
        pa = [[-1] * m for _ in range(n)]

        def bfs(root):
            q = deque([root])
            while q:
                u = q.popleft()
                for v in g[u]:
                    if depth[v] > depth[u] + 1:
                        depth[v] = depth[u] + 1
                        q.append(v)
                        pa[v][0] = u
                        for k in range(m - 1):
                            pa[v][k + 1] = pa[pa[v][k]][k]

        bfs(root)
        self.depth = depth
        self.pa = pa

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(k.bit_length()):
            if (k >> i) & 1:  # k 二进制从低到高第 i 位是 1
                node = self.pa[node][i]
                if node == -1:
                    break
        return node

    # 返回 x 和 y 的最近公共祖先（节点编号从 0 开始）
    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        # 使 y 和 x 在同一深度
        y = self.get_kth_ancestor(y, self.depth[y] - self.depth[x])
        if y == x:
            return x
        for i in range(len(self.pa[x]) - 1, -1, -1):
            px, py = self.pa[x][i], self.pa[y][i]
            if px != py:
                x, y = px, py  # 同时上跳 2**i 步
        return self.pa[x][0]


# 节点编号先离散化，离散化之后节点从0开始，不需要哨兵
n = int(input())
idx = set()
root = -1
edges = []
for _ in range(n):
    a, b = map(int, input().split())
    idx.add(a)
    if b == -1:
        root = a
    else:
        idx.add(b)
        edges.append((a, b))
idx = {e: i for i, e in enumerate(sorted(idx))}
edges = [(idx[a], idx[b]) for a, b in edges]
root = idx[root]
lca = TreeAncestor(len(idx), edges)

for _ in range(int(input())):
    x, y = map(int, input().split())
    x, y = idx[x], idx[y]
    t = lca.get_lca(x, y)
    if t == x:
        print(1)
    elif t == y:
        print(2)
    else:
        print(0)
