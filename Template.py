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

"""
IO专用
"""
RI = lambda: map(int, sys.stdin.buffer.readline().split())
RS = lambda: map(bytes.decode, sys.stdin.buffer.readline().strip().split())
RILST = lambda: list(RI())

"""
并查集 非递归版本
"""


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
        if self.father[x] != x:
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
            self.father[x] = x


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

    # n = 5
    # edges = [[]]
    # g = [{} for _ in range(n)]
    # for x, y, w in edges:
    #     if w != -1:
    #         g[x][y] = w
    #         g[y][x] = w

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

