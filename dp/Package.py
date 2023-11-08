# # 背包问题
# # 01背包
# ## 记忆化搜索
from collections import Counter, defaultdict
from typing import List

# 递推优化
n, m = map(int, input().split())
a = []
for _ in range(n):
    v, w = map(int, input().split())
    a.append((v, w))

f = [0] * (1 + m)
# v,w 体积，价值
for v, w in a:
    for j in range(m, v - 1, -1):
        f[j] = max(f[j], f[j - v] + w)
print(f[-1])

# 完全背包（无限选择）
n, m = map(int, input().split())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))

f = [0] * (m + 1)
for v, w in a:
    # 与01背包的不同点，这里是正序遍历j，因为f[i][j]是从f[i][j - v]转移来
    for j in range(v, m + 1):
        f[j] = max(f[j], f[j - v] + w)
print(f[-1])

# 多重背包求最大价值
n, m = map(int, input().split())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))

# 多重背包求最大价值-朴素 转换成01背包求解
f = [0] * (m + 1)
for v, w, s in a:
    for j in range(m, v - 1, -1):
        for k in range(s + 1):
            if k * v > j:
                break
            f[j] = max(f[j], f[j - k * v] + k * w)
print(f[-1])

# 多重背包求最大价值-二进制优化
n, m = map(int, input().split())
a = []

for _ in range(n):
    v, w, s = map(int, input().split())
    # 二进制拆分
    i = 1
    while i <= s:
        a.append((v * i, w * i))
        s -= i
        i <<= 1
    if s:
        a.append((s * v, s * w))

f = [0] * (m + 1)
for v, w in a:
    for j in range(m, v - 1, -1):
        f[j] = max(f[j - v] + w, f[j])
print(f[-1])

# 多重背包求最大价值-单调队列优化
n, m = map(int, input().split())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))

f = [0] * (m + 1)
q = [0] * (m + 1)
for v, w, s in a:
    g = f[:]
    for j in range(v):  # 与v同余的在一个单调队列中，单调队列元素值递减
        hh, tt = 0, -1
        for k in range(j, m + 1, v):  # 每次滑动v
            # f[k] = g[k]
            if hh <= tt and k - s * v > q[hh]:  # 如果k/v - q[0] > s，代表已经滑动了超过s个v
                hh += 1
            if hh <= tt:  # 更新f
                f[k] = max(g[k], g[q[hh]] + (k - q[hh]) // v * w)  # 从最大值（队首）转移过来，再装k-q[0]个物品
            while hh <= tt and g[q[tt]] - (q[tt] - j) // v * w <= g[k] - (k - j) // v * w:
                # while hh <= tt and g[q[tt]] + (k - q[tt]) // v * w <= g[k]:
                tt -= 1
            tt += 1
            q[tt] = k
print(f[-1])


# 多重背包求方案数
def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
    MOD = 10 ** 9 + 7
    total = sum(nums)
    if l > total:
        return 0

    m = min(r, total)
    cnt = Counter(nums)
    f = [cnt[0] + 1] + [0] * m  # f[i]代表体积恰好为i的方案数，体积为0归在一起
    del cnt[0]

    for v, s in cnt.items():
        g = f[:]
        for j in range(v, m + 1):
            g[j] += g[j - v]
            if j >= (s + 1) * v:
                g[j] -= f[j - (s + 1) * v]
            g[j] %= MOD
        f = g
    return sum(f[l:]) % MOD


# 混合背包问题
# 有 N种物品和一个容量是 V的背包。
# 物品一共有三类：
# 第一类物品只能用1次（01背包）；
# 第二类物品可以用无限次（完全背包）；
# 第三类物品最多只能用 si次（多重背包）；
# 每种体积是 vi，价值是 wi
# 求解将哪些物品装入背包，可使物品体积总和不超过背包容量，且价值总和最大。
# 输出最大价值。
n, m = map(int, input().split())
a = []
for _ in range(n):
    v, w, s = map(int, input().split())
    if s == -1:
        # 01背包
        a.append((-1, v, w))
    elif s == 0:
        # 完全背包
        a.append((0, v, w))
    else:
        # 多重背包转01背包
        k = 1
        while k <= s:
            a.append((-1, v * k, w * k))
            s -= k
            k <<= 1
        if s:
            a.append((-1, v * s, w * s))

f = [0] * (m + 1)
for t, v, w in a:
    if t == -1:
        # 01 背包
        for j in range(m, v - 1, -1):
            f[j] = max(f[j], f[j - v] + w)
    else:
        # 完全背包
        for j in range(v, m + 1):
            f[j] = max(f[j], f[j - v] + w)
print(f[-1])

# 二维费用背包
# 有 N件物品和一个容量是 V的背包，背包能承受的最大重量是 M
# 每件物品只能用一次。体积是 vi，重量是 mi，价值是 wi
# 求解将哪些物品装入背包，可使物品总体积不超过背包容量，总重量不超过背包可承受的最大重量，且价值总和最大。
# 输出最大价值。
N, V, M = map(int, input().split())
a = []
for _ in range(N):
    v, m, w = map(int, input().split())
    a.append((v, m, w))

f = [[0] * (M + 1) for _ in range(V + 1)]
for v, m, w in a:
    for j in range(V, v - 1, -1):
        for k in range(M, m - 1, -1):
            f[j][k] = max(f[j][k], f[j - v][k - m] + w)
print(f[-1][-1])

# 分组背包
# 有 N组物品和一个容量是 V的背包。
# 每组物品有若干个，同一组内的物品最多只能选一个。
# 每件物品的体积是 vij，价值是 wij，其中 i是组号，j是组内编号。
# 求解将哪些物品装入背包，可使物品总体积不超过背包容量，且总价值最大。
# 输出最大价值。

N, V = map(int, input().split())
a = []
for _ in range(N):
    b = []
    for _ in range(int(input())):
        b.append(list(map(int, input().split())))
    a.append(b)
# 记忆化搜索
# from functools import lru_cache
# @lru_cache(None)
# def dfs(i, V):
#     if i < 0:
#         return 0
#     ans = dfs(i - 1, V)
#     for v, w in a[i]:
#         if v <= V:
#             ans = max(ans, dfs(i - 1, V - v) + w)
#     return ans
# print(dfs(N - 1, V))

# 二维
# f = [[0]*(V + 1) for _ in range(N + 1)]
# for i in range(1,N + 1):
#     for j in range(1,V + 1):
#         f[i][j] = f[i - 1][j]
#         for k,(v,w) in enumerate(a[i - 1]):
#             if v <= j:
#                 f[i][j] = max(f[i][j],f[i - 1][j - v] + w)
# print(f[-1][-1])

# 一维
f = [0] * (V + 1)
for i in range(1, N + 1):
    for j in range(V, 0, -1):
        for k, (v, w) in enumerate(a[i - 1]):
            if v <= j:
                f[j] = max(f[j], f[j - v] + w)
print(f[-1])

# 有依赖的背包问题
N, V = map(int, input().split())
from collections import defaultdict
g = defaultdict(list)
a = [-1]
for i in range(1,N + 1):
    v, w, p = map(int, input().split())
    if p == -1:
        root = i
    else:
        g[p].append(i)
    a.append((v, w, p))

f = [[0]*(V + 1) for _ in range(N + 1)]

def dfs(u):
    v, w, p = a[u]
    for i in range(v,V + 1):
        f[u][i] = w
    for son in g[u]:
        dfs(son)
        for j in range(V, v - 1, -1):
            for k in range(j - v + 1):
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])
dfs(root)
print(f[root][V])


# 01背包求方案数
n, m = map(int, input().split())
a = []
for _ in range(n):
    v, w = map(int, input().split())
    a.append((v, w))

MOD = 10**9 + 7
f = [0] * (1 + m)
c = [1] * (1 + m)
# v,w 体积，价值
for v, w in a:
    for j in range(m, v - 1, -1):
        if f[j - v] + w > f[j]:
            f[j] = f[j - v] + w
            c[j] = c[j - v]
        elif f[j - v] + w == f[j]:
            c[j] += c[j - v]
            c[j] %= MOD
print(c[-1])

# 01背包求具体方案
n, m = map(int, input().split())
a = []
for _ in range(n):
    v, w = map(int, input().split())
    a.append((v, w))

f = [[0] * (1 + m) for _ in range(2 + n)]
p = [[0]*(1 + m) for _ in range(1 + n)]
for i in range(n,0,-1): # 逆序取物
    v,w = a[i - 1]
    for j in range(m + 1):
        f[i][j] = f[i + 1][j]
        p[i][j] = j
        if j >= v:
            f[i][j] = max(f[i][j],f[i + 1][j - v] + w)
        if j >= v and f[i + 1][j - v] + w == f[i][j]:
            p[i][j] = j - v
j = m
ans = []
for i in range(1,n + 1):
    if p[i][j] < j:
        ans.append(str(i))
    j = p[i][j]
print(' '.join(ans))