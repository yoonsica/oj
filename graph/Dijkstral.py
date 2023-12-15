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


# 最短路计数

from collections import defaultdict
from heapq import heappop, heappush
from math import inf

mod = 100003
n, m = map(int, input().split())
g = defaultdict(list)

for _ in range(m):
    u, v = map(int, input().split())
    g[u].append(v)
    g[v].append(u)

dis = [inf] * (n + 1)
dis[1] = 0
cnt = [0] * (n + 1)
cnt[1] = 1
q = [(0, 1)]
while q:
    d, u = heappop(q)
    if dis[u] < d:
        continue
    for v in g[u]:
        if dis[v] > d + 1:
            dis[v] = d + 1
            heappush(q, (d + 1, v))
            cnt[v] = cnt[u]
        elif dis[v] == d + 1:
            cnt[v] = (cnt[v] + cnt[u]) % mod

for i in range(1, n + 1):
    print(cnt[i])


from heapq import heappop, heappush
from math import inf
from collections import defaultdict
# 最短路+次短路
def dj():
    dis = [[inf,inf] for _ in range(n + 1)]
    cnt = [[0,0] for _ in range(n + 1)]
    dis[s][0] = 0
    cnt[s][0] = 1
    q = [(0,s,0)] # d,u,type( 0:最短路 1:次短路)
    while q:
        d,u,t = heappop(q)
        if dis[u][t] < d:
            continue
        for v,w in g[u]:
            # 尝试更新最短路
            if dis[v][0] > d + w:
                cnt[v] = [cnt[u][t],cnt[v][0]]
                dis[v][1] = dis[v][0]
                dis[v][0] = d + w
                heappush(q,(dis[v][0],v,0))
                heappush(q,(dis[v][1],v,1))
            elif dis[v][0] == d + w:
                cnt[v][0] += cnt[u][t]
            # 尝试更新次短路
            elif dis[v][1] > d + w:
                cnt[v][1] = cnt[u][t]
                dis[v][1] = d + w
                heappush(q,(dis[v][1],v,1))
            elif dis[v][1] == d + w:
                cnt[v][1] += cnt[u][t]

    if dis[f][1] - dis[f][0] == 1:
        return sum(cnt[f])
    else:
        return cnt[f][0]

for _ in range(int(input())):
    n,m = map(int,input().split())
    g = defaultdict(list)
    for _ in range(m):
        a,b,l = map(int,input().split())
        g[a].append((b,l))
    s,f = map(int,input().split())
    print(dj())
