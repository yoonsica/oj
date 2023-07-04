"""
最小生成树
"""

# edges = [(u,v,cost)]
from UnionFind import UnionFind


def kruskal(edges):
    edges.sort(key=lambda x: x[2])
    uf = UnionFind.UnionFind()
    res = 0
    for (u, v, cost), e in enumerate(edges):
        if not uf.is_connected(u, v):
            uf.union(u, v)
            res += cost
    return res
