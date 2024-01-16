# tarjan求LCA
# 快读板子
import sys
from collections import defaultdict

RI = lambda: int(sys.stdin.readline().strip())
RS = lambda: sys.stdin.readline().strip()
RII = lambda: map(int, sys.stdin.readline().strip().split())
RILIST = lambda: list(RII())

n, m = RII()
g = defaultdict(list)
for _ in range(n - 1):
    a, b, k = RII()
    g[a].append((b, k))
    g[b].append((a, k))

# tarjan是离线求lca的方法，记录查询
queries = defaultdict(list)
for i in range(m):
    a, b = RII()
    # 这里要注意特判两个点相同的情况
    if a != b:
        queries[a].append((b, i))
        queries[b].append((a, i))
ans = [0] * m
st = [0] * (1 + n)  # 节点编号从1开始的
dis = [0] * (1 + n)


def dfs(u, fa):
    for v, w in g[u]:
        if v != fa:
            dis[v] = dis[u] + w
            dfs(v, u)


# 并查集板子
class UnionFind:
    def __init__(self, n):
        self.fa = list(range(n))

    def find(self, x):
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a != b:
            self.fa[a] = b

    def is_same(self, x, y):
        return self.find(x) == self.find(y)


def tarjan(u):
    st[u] = 1  # 正在搜索的点为1，没搜索到的为0，回溯过的为2
    for v, w in g[u]:
        if not st[v]:
            tarjan(v)
            # v回溯过了，合并
            uf.union(v, u)
    # 更新答案，如果v回溯过，v的祖先就是u，v的lca
    for v, idx in queries[u]:
        if st[v] == 2:
            lca = uf.find(v)
            ans[idx] = dis[u] + dis[v] - 2 * dis[lca]
    st[u] = 2  # 回溯过改为2


dfs(1, -1)
uf = UnionFind(n + 1)
tarjan(1)
for x in ans:
    print(x)
