"""
迪杰斯特拉
"""
import heapq
from collections import deque
from math import inf


# n = 5
# edges = [[]]
# g = [{} for _ in range(n)]
# for x, y, w in edges:
#     if w != -1:
#         g[x][y] = w
#         g[y][x] = w

# 朴素
def dj(s, e, g, n):
    dis = [inf] * n
    dis[s] = 0
    vis = [False]*n
    while True:
        x = -1
        for i in range(n):
            if not vis[i] and (x < 0 or dis[i] < dis[x]):
                x = i
        if x < 0 or dis[x] == inf:
            return -1
        if x == e:
            return dis[x]
        vis[x] = True
        for y, w in enumerate(g[x]):
            dis[y] = min(dis[y], dis[x] + w)


# 堆优化
def dj_heap(s, e, g, n):
    dis = [inf] * n
    dis[s] = 0
    q = [(0, s)]
    while q:
        d, v = heapq.heappop(q)
        if dis[v] < d:
            continue
        if v == e:
            return dis[v]
        for a, b in g[v].items():
            if d + b < dis[a]:
                dis[a] = d + b
                heapq.heappush(q, (d + b, a))
    return inf
