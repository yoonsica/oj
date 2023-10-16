"""
引用
"""
import sys
import random
from types import GeneratorType
from itertools import *
import io, os
from typing import List
from bisect import *
from collections import *
from contextlib import redirect_stdout
from itertools import *
from array import *
from functools import lru_cache, reduce
from heapq import *
from math import *
import heapq
import sys
from collections import defaultdict

from collections import Counter
"""
IO专用
"""
RI = lambda: map(int, sys.stdin.buffer.readline().split())
RS = lambda: map(bytes.decode, sys.stdin.buffer.readline().strip().split())
RILST = lambda: list(RI())


# 突破递归深度限制
# sys.setrecursionlimit(n)

"""
并查集
"""
class UnionFind:
    def __init__(self,n):
        self.fa = list(range(n * n))

    def find(self, x: int) -> int:
        fa = self.fa
        if fa[x] != x:
                fa[x] = self.find(fa[x])
        return fa[x]

    def union(self,x: int, y: int):
        self.fa[self.find(x)] = self.find(y)

    def is_conn(self,x,y):
        return self.find(x) == self.find(y)

"""
字典树
"""
class Trie:

    def __init__(self):
        self.child = {}
        self.isEnd = False

    def insert(self, word: str) -> None:
        root = self
        for c in word:
            if c not in root.child:
                root.child[c] = Trie()
            root = root.child[c]
        root.isEnd = True

    def search(self, word: str) -> bool:
        root = self
        for c in word:
            if c not in root.child:
                return False
            root = root.child[c]
        return root.isEnd

    def startsWith(self, prefix: str) -> bool:
        root = self
        for c in prefix:
            if c not in root.child:
                return False
            root = root.child[c]
        return True


"""
最短路
"""


class Dijkstral:
    # 朴素
    def dj(s, e, g, n):
        dis = [float('inf')] * n
        dis[s] = 0
        vis = [False] * n
        prev = [-1] * n  # 用来求最短路径
        while True:
            x = -1
            for i in range(n):
                if not vis[i] and (x < 0 or dis[i] < dis[x]):
                    x = i
            if x < 0 or dis[x] == float('inf'):
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
        dis = [float('inf')] * n
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
        return float('inf')


class KMP:
    def strStr(self, haystack: str, needle: str) -> int:
        def get_next(p):
            k, j, n = -1, 0, len(p)
            next = [-1] * n
            while j < n - 1:
                if k == -1 or p[j] == p[k]:
                    k += 1
                    j += 1
                    next[j] = next[k] if p[j] == p[k] else k
                else:
                    k = next[k]
            return next

        next = get_next(needle)
        i = j = 0
        while i < len(haystack) and j < len(needle):
            if j == -1 or haystack[i] == needle[j]:
                j += 1
                i += 1
            else:
                j = next[j]
            if j == len(needle):
                return i - j
        return -1


# 快速查询
# 第k个元素=k - 1个位置上的元素
def quick_Select(arr, s, e, k):
    l, r = s, e
    pivot = arr[s]
    while l < r:
        while l < r and arr[r] >= pivot:
            r -= 1
        if l < r:
            arr[l] = arr[r]
            l += 1
        while l < r and arr[l] <= pivot:
            l += 1
        if l < r:
            arr[r] = arr[l]
            r -= 1
    arr[l] = pivot
    if l < k - 1:
        quick_Select(arr, l + 1, e, k)
    if l > k - 1:
        quick_Select(arr, s, l - 1, k)


"""
最长递增子序列LIS
"""


def lengthOfLIS(self, nums: List[int]) -> int:
    g = []
    for x in nums:
        i = bisect_left(g, x)
        if i == len(g):
            g.append(x)
        else:
            g[i] = x
    return len(g)


def lengthOfLIS1(self, nums: List[int]) -> int:
    ng = 0
    for x in nums:
        i = bisect_left(nums, x, 0, ng)
        if i == ng:
            ng += 1
        nums[i] = x
    return ng


# 埃筛
def ai_primes(n):
    if n < 2:
        return 0
    is_prime = [1] * n
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n, i):
                is_prime[j] = 0
    return is_prime


# 线性筛/欧拉筛
def ol_primes(n):
    if n < 2:
        return 0
    is_prime = [1] * n
    is_prime[0] = is_prime[1] = 0
    primes = []
    for i in range(2, n):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p >= n:
                break
            is_prime[i * p] = 0
            if i % p == 0:
                break
    return primes


# 分解质因数
def get_prime_factors(x):
    ans = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            # f(d,i) d是质因数
            ans.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        ans.append(x)  # f(x,i) x也是质因数
    return ans

# 动态开点线段树 单点更新，区间查询（求和）
class Node:
    __slots__ = ['val', 'left', 'right']

    def __init__(self, left=None, right=None, val=0) -> None:
        self.left, self.right, self.val = left, right, val

# 线段树
class SegmentTree:
    def __init__(self):
        self.root = Node()

    # 单点增加val
    def update_index(self, node, s, e, index, val):
        # 动态开点
        if not node.left:
            node.left = Node()
        if not node.right:
            node.right = Node()
        if s == e:
            node.val = val
            return node
        mid = (s + e) >> 1
        if index <= mid:
            self.update_index(node.left, s, mid, index, val)
        if index > mid:
            self.update_index(node.right, mid + 1, e, index, val)
        node.val = node.left.val + node.right.val

    # 区间更新
    def update(self, node, s, e, l, r, val):
        # 动态开点
        if not node.left:
            node.left = Node()
        if not node.right:
            node.right = Node()
        if l <= s and e <= r:
            node.val = val
            return node
        mid = (s + e) >> 1
        if l <= mid:
            self.update(node.left, s, mid, l, r, val)
        if r > mid:
            self.update(node.right, mid + 1, e, l, r, val)
        node.val = node.left.val + node.right.val


    def query(self, node, s, e, l, r):
        if l <= s and e <= r:
            return node.val
        mid = (s + e) >> 1
        ans = 0
        if l <= mid:
            ans += self.query(node.left, s, mid, l, r)
        if r > mid:
            ans += self.query(node.right, mid + 1, e, l, r)
        return ans

# 树状数组
class BIT:
    def __init__(self, nums):
        self.tree = [0] * (len(nums) + 1)
        for i, x in enumerate(nums, 1):
            self.add(i, x)

    def add(self, i, v):
        while i < len(self.tree):
            self.tree[i] += v
            i += i & -i

    def query(self, i):
        ans = 0
        while i:
            ans += self.tree[i]
            i &= i - 1
        return ans

# sortedList
class SortedList:
    def __init__(self, iterable=[], _load=200):
        """Initialize sorted list instance."""
        values = sorted(iterable)
        self._len = _len = len(values)
        self._load = _load
        self._lists = _lists = [values[i:i + _load] for i in range(0, _len, _load)]
        self._list_lens = [len(_list) for _list in _lists]
        self._mins = [_list[0] for _list in _lists]
        self._fen_tree = []
        self._rebuild = True

    def _fen_build(self):
        """Build a fenwick tree instance."""
        self._fen_tree[:] = self._list_lens
        _fen_tree = self._fen_tree
        for i in range(len(_fen_tree)):
            if i | i + 1 < len(_fen_tree):
                _fen_tree[i | i + 1] += _fen_tree[i]
        self._rebuild = False

    def _fen_update(self, index, value):
        """Update `fen_tree[index] += value`."""
        if not self._rebuild:
            _fen_tree = self._fen_tree
            while index < len(_fen_tree):
                _fen_tree[index] += value
                index |= index + 1

    def _fen_query(self, end):
        """Return `sum(_fen_tree[:end])`."""
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        x = 0
        while end:
            x += _fen_tree[end - 1]
            end &= end - 1
        return x

    def _fen_findkth(self, k):
        """Return a pair of (the largest `idx` such that `sum(_fen_tree[:idx]) <= k`, `k - sum(_fen_tree[:idx])`)."""
        _list_lens = self._list_lens
        if k < _list_lens[0]:
            return 0, k
        if k >= self._len - _list_lens[-1]:
            return len(_list_lens) - 1, k + _list_lens[-1] - self._len
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        idx = -1
        for d in reversed(range(len(_fen_tree).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(_fen_tree) and k >= _fen_tree[right_idx]:
                idx = right_idx
                k -= _fen_tree[idx]
        return idx + 1, k

    def _delete(self, pos, idx):
        """Delete value at the given `(pos, idx)`."""
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len -= 1
        self._fen_update(pos, -1)
        del _lists[pos][idx]
        _list_lens[pos] -= 1

        if _list_lens[pos]:
            _mins[pos] = _lists[pos][0]
        else:
            del _lists[pos]
            del _list_lens[pos]
            del _mins[pos]
            self._rebuild = True

    def _loc_left(self, value):
        """Return an index pair that corresponds to the first position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        lo, pos = -1, len(_lists) - 1
        while lo + 1 < pos:
            mi = (lo + pos) >> 1
            if value <= _mins[mi]:
                pos = mi
            else:
                lo = mi

        if pos and value <= _lists[pos - 1][-1]:
            pos -= 1

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value <= _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def _loc_right(self, value):
        """Return an index pair that corresponds to the last position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        pos, hi = 0, len(_lists)
        while pos + 1 < hi:
            mi = (pos + hi) >> 1
            if value < _mins[mi]:
                hi = mi
            else:
                pos = mi

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value < _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def add(self, value):
        """Add `value` to sorted list."""
        _load = self._load
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len += 1
        if _lists:
            pos, idx = self._loc_right(value)
            self._fen_update(pos, 1)
            _list = _lists[pos]
            _list.insert(idx, value)
            _list_lens[pos] += 1
            _mins[pos] = _list[0]
            if _load + _load < len(_list):
                _lists.insert(pos + 1, _list[_load:])
                _list_lens.insert(pos + 1, len(_list) - _load)
                _mins.insert(pos + 1, _list[_load])
                _list_lens[pos] = _load
                del _list[_load:]
                self._rebuild = True
        else:
            _lists.append([value])
            _mins.append(value)
            _list_lens.append(1)
            self._rebuild = True

    def discard(self, value):
        """Remove `value` from sorted list if it is a member."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_right(value)
            if idx and _lists[pos][idx - 1] == value:
                self._delete(pos, idx - 1)

    def remove(self, value):
        """Remove `value` from sorted list; `value` must be a member."""
        _len = self._len
        self.discard(value)
        if _len == self._len:
            raise ValueError('{0!r} not in list'.format(value))

    def pop(self, index=-1):
        """Remove and return value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        value = self._lists[pos][idx]
        self._delete(pos, idx)
        return value

    def bisect_left(self, value):
        """Return the first index to insert `value` in the sorted list."""
        pos, idx = self._loc_left(value)
        return self._fen_query(pos) + idx

    def bisect_right(self, value):
        """Return the last index to insert `value` in the sorted list."""
        pos, idx = self._loc_right(value)
        return self._fen_query(pos) + idx

    def count(self, value):
        """Return number of occurrences of `value` in the sorted list."""
        return self.bisect_right(value) - self.bisect_left(value)

    def __len__(self):
        """Return the size of the sorted list."""
        return self._len

    def __getitem__(self, index):
        """Lookup value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        return self._lists[pos][idx]

    def __delitem__(self, index):
        """Remove value at `index` from sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        self._delete(pos, idx)

    def __contains__(self, value):
        """Return true if `value` is an element of the sorted list."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_left(value)
            return idx < len(_lists[pos]) and _lists[pos][idx] == value
        return False

    def __iter__(self):
        """Return an iterator over the sorted list."""
        return (value for _list in self._lists for value in _list)

    def __reversed__(self):
        """Return a reverse iterator over the sorted list."""
        return (value for _list in reversed(self._lists) for value in reversed(_list))

    def __repr__(self):
        """Return string representation of sorted list."""
        return 'SortedList({0})'.format(list(self))

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
                f[k] = max(f[k], g[q[hh]] + (k - q[hh]) // v * w)  # 从最大值（队首）转移过来，再装k-q[0]个物品
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