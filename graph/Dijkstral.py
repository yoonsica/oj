"""
迪杰斯特拉
"""
import heapq
from math import inf
class Dikstral:
    # 朴素
    def dj(s, e, g, n):
        dis = [inf] * n
        dis[s] = 0
        vis = [False] * n
        prev = [-1] * n  # 用来求最短路径
        while True:
            x = -1
            for i in range(n):
                if not vis[i] and (x < 0 or dis[i] < dis[x]):
                    x = i
            if x < 0 or dis[x] == inf:
                return -1
            if x == e:
                # 最短路径 get_path(prev,e)
                return dis[x]
            vis[x] = True
            for y, w in enumerate(g[x]):
                # dis[y] = min(dis[y], dis[x] + w)
                if dis[x] + w < dis[y]:
                    prev[y] = x
                    dis[y] = dis[x] + w

    def dj_get_path(prev, e):
        path = [e]
        t = prev[e]
        while t != -1:
            path.append(t)
            t = prev[t]
        return path[::-1]

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
