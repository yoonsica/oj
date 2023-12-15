from math import inf


def spfa(s, g, n):
    dis = [inf] * (n + 1)
    st = [0] * (n + 1)
    st[s] = True
    q = [s]
    dis[s] = 0
    while q:
        u = q.pop()
        st[u] = False
        for v, w in g[u]:
            if dis[v] > dis[u] + w:
                dis[v] = dis[u] + w
                if not st[v]:
                    st[v] = True
                    q.append(v)
    return dis

# spfa找负环
def spfa_circle(g,n):
    # 找负环，所有节点全部入队，距离为0，超级源点思想
    dis = [0] * (n + 1)
    st = [0] + [1]*n
    q = list(range(1,n + 1))
    cnt = [0]*(n + 1) # 记录每个节点的最短路径经过的边数量，一旦等于n就有环了
    while q:
        u = q.pop()
        st[u] = False
        for v, w in g[u]:
            if dis[v] > dis[u] + w:
                dis[v] = dis[u] + w
                cnt[v] = cnt[u] + 1
                if cnt[v] == n:
                    return True
                if not st[v]:
                    st[v] = True
                    q.append(v)
    return False