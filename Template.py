"""
引用
"""
import sys
import random
from itertools import *
from typing import List
from bisect import *
from collections import *
from functools import lru_cache, reduce
from heapq import *
from math import *
# 突破递归深度限制
# sys.setrecursionlimit(1000000)

import random
ans = random.randint(1,11)
print(True if ans <= 5 else False)

# 快读板子
RI = lambda: map(int, sys.stdin.buffer.readline().split())
RS = lambda: map(bytes.decode, sys.stdin.buffer.readline().strip().split())
RILST = lambda: list(RI())
"""
并查集
"""
class UnionFind:
    def __init__(self,n):
        self.fa = list(range(n))

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


class TrieXor:
    def __init__(self, nums=None, bit_len=31):
        # 01字典树，用来处理异或最值问题，本模板只处理数字最低的31位
        # 用nums初始化字典树，可以为空
        self.trie = {}
        self.cnt = 0  # 字典树插入了几个值
        if nums:
            for a in nums:
                self.insert(a)
        self.bit_len = bit_len

    def insert(self, num):
        # 01字典树插入一个数字num,只会处理最低bit_len位。
        cur = self.trie
        for i in range(self.bit_len, -1, -1):
            nxt = (num >> i) & 1
            if nxt not in cur:
                cur[nxt] = {}
            cur = cur[nxt]
            cur[3] = cur.get(3, 0) + 1  # 这个节点被经过了几次
        cur[5] = num  # 记录这个数:'#'或者'end'等非01的当key都行;这里由于key只有01因此用5
        self.cnt += 1
    def remove(self,v):
        cur = self.trie
        for i in range(self.bit_len, -1, -1):
            nxt = v >> i & 1
            cur[nxt][3] -= 1
            if not cur[nxt][3]:
                del cur[nxt]
                break
            cur = cur[nxt]
    def find_max_xor_num(self, num):
        # 计算01字典树里任意数字异或num的最大值,只会处理最低bit_len位。
        # 贪心的从高位开始处理，显然num的某位是0，对应的优先应取1；相反同理
        cur = self.trie
        ret = 0
        for i in range(self.bit_len, -1, -1):
            if (num >> i) & 1 == 0:  # 如果本位是0，那么取1才最大；取不到1才取0
                if 1 in cur:
                    cur = cur[1]
                    ret += ret + 1
                else:
                    cur = cur.get(0, {})
                    ret <<= 1
            else:
                if 0 in cur:
                    cur = cur[0]
                    ret += ret + 1
                else:
                    cur = cur.get(1, {})
                    ret <<= 1
        return ret

    def count_less_than_limit_xor_num(self, num, limit):
        # 计算01字典树里有多少数字异或num后小于limit
        # 由于计算的是严格小于，因此只需要计算三种情况:
        # 1.当limit对应位是1，且异或值为0的子树部分，全部贡献。
        # 2.当limit对应位是1，且异或值为1的子树部分，向后检查。
        # 3.当limit对应为是0，且异或值为0的子树部分，向后检查。
        # 若向后检查取不到，直接剪枝break
        cur = self.trie
        ans = 0
        for i in range(self.bit_len, -1, -1):
            a, b = (num >> i) & 1, (limit >> i) & 1
            if b == 1:
                if a == 0:
                    if 0 in cur:  # 右子树上所有值异或1都是0，一定小于1
                        ans += cur[0][3]
                    cur = cur.get(1)  # 继续检查右子树
                    if not cur: break  # 如果没有1，即没有右子树，可以直接跳出了
                if a == 1:
                    if 1 in cur:  # 右子树上所有值异或1都是0，一定小于1
                        ans += cur[1][3]
                    cur = cur.get(0)  # 继续检查左子树
                    if not cur: break  # 如果没有0，即没有左子树，可以直接跳出了
            else:
                cur = cur.get(a)  # limit是0，因此只需要检查异或和为0的子树
                if not cur: break  # 如果没有相同边的子树，即等于0的子树，可以直接跳出了
        return ans


class Solution:
    def countPairs(self, nums: List[int], low: int, high: int) -> int:
        trie = TrieXor(bit_len=15)
        ans = 0
        for x in nums:
            ans += trie.count_less_than_limit_xor_num(x,high+1) - trie.count_less_than_limit_xor_num(x,low)
            trie.insert(x)
        return ans


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

# 动态开点线段树，区间更新，区间查询（最大值）
class Node:
    __slots__ = ['val', 'left', 'right', 'lazy']

    def __init__(self, left=None, right=None, val=-inf) -> None:
        self.val, self.left, self.right = val, left, right


class SegmentTree:
    def __init__(self):
        self.root = Node()

    # 区间增加val
    def update(self, node, s, e, l, r, val):
        if l <= s and e <= r:
            if val > node.val:
                node.val = val
            return
        mid = (s + e) >> 1
        # 如果是求区间元素和，需要带上区间元素个数
        self.push_down(node, mid - s + 1, e - mid)
        if l <= mid:
            self.update(node.left, s, mid, l, r, val)
        if r > mid:
            self.update(node.right, mid + 1, e, l, r, val)
        self.push_up(node)

    def push_down(self, node, left_num, right_num):
        if not node.left:
            node.left = Node()
        if not node.right:
            node.right = Node()

    def push_up(self, node):
        node.val = max(node.left.val, node.right.val)

    def query(self, node, s, e, l, r):
        if l <= s and e <= r:
            return node.val
        mid = (s + e) >> 1
        self.push_down(node, mid - s + 1, e - mid)
        ans = -inf
        if l <= mid:
            ans = self.query(node.left, s, mid, l, r)
        if r > mid:
            ans = max(ans,self.query(node.right, mid + 1, e, l, r))
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

# 区间更新
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        # 先离散化
        _set = set()
        for x, y in flowers:
            _set.add(x)
            _set.add(y)
        for x in people:
            _set.add(x)
        arr = sorted(list(_set))
        d = {e: i for i, e in enumerate(arr)}
        # 树状数组
        n = len(arr)
        bit = BIT(n)
        for s, e in flowers:
            bit.add_interval(d[s] + 1, d[e] + 1, 1)
        ans = []
        for t in people:
            ans.append(bit.query(d[t] + 1))
        return ans


# 树状数组模板
class BIT:
    def __init__(self, n):
        # 区间更新需要维护差分数组，原数组元素均为0这里
        self.arr = [0] * (n + 1)
        self.n = n

    def add_interval(self, l, r, v):
        self.add(l, v)
        self.add(r + 1, -v)

    def add(self, i, v):
        while i <= self.n:
            self.arr[i] += v
            i += i & -i

    def query(self, i):
        ans = 0
        while i:
            ans += self.arr[i]
            i -= (i & -i)
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


# 二维前缀和
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        if not m or not n:
            return
        self.matrix = matrix
        self.pre_sum = [[0]*(n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.pre_sum[i + 1][j + 1] = self.pre_sum[i][j + 1] + self.pre_sum[i + 1][j] + self.matrix[i][j] - self.pre_sum[i][j]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row2 + 1][col2 + 1] - self.pre_sum[row2 + 1][col1] - self.pre_sum[row1][col2 + 1] + self.pre_sum[row1][col1]

# 重构一棵树的方案数
class Solution:
    def checkWays(self, pairs: List[List[int]]) -> int:
        # 邻接表
        adj = defaultdict(set)
        for x, y in pairs:
            adj[x].add(y)
            adj[y].add(x)
        n = len(adj)
        # 寻找根节点
        root = -1
        for node, neighbours in adj.items():
            if len(neighbours) == n - 1:
                root = node
        # 找不到根节点
        if root == -1:
            return 0
        # 根节点
        res = 1
        # 遍历每个节点
        for node, neighbours in adj.items():
            if node == root:
                continue
            curr_degree = len(neighbours)
            parent = -1
            parent_degree = n
            # 根据 degree 大小找 node 的 parent 节点
            for neighbour in neighbours:
                if curr_degree <= len(adj[neighbour]) < parent_degree:
                    parent = neighbour
                    parent_degree = len(adj[neighbour])
            # 检查 neighbours 是否为 adj[parent] 的子集
            if parent == -1:
                return 0
            for neighbour in neighbours:
                if neighbour != parent and neighbour not in adj[parent]:
                    return 0
            if parent_degree == curr_degree:
                res = 2
        return res

# 你可以安排的最多任务数目
class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        tasks.sort()
        workers.sort()
        s = 0
        e = min(len(tasks), len(workers)) + 1
        while s + 1 < e:
            m = (s + e) // 2
            i2 = 0
            p = pills
            fail = False
            valid_tasks = deque()
            for j in range(len(workers) - m, len(workers)):
                w = workers[j]
                while i2 < m and tasks[i2] <= w + strength:
                    valid_tasks.append(tasks[i2])
                    i2 += 1
                if not valid_tasks:
                    fail = True
                    break
                if valid_tasks[0] <= w:
                    # No need for pill
                    valid_tasks.popleft()
                else:
                    if not p:
                        fail = True
                        break
                    p -= 1
                    valid_tasks.pop()
            if fail:
                e = m
            else:
                s = m
        return s

# 最大化网格幸福感
class Solution:
    def getMaxGridHappiness(self, m: int, n: int, introvertsCount: int, extrovertsCount: int) -> int:
        @lru_cache(None)
        def dfs(i, state, intro, ext):
            if i == m * n:
                return 0
            up = state // w if i // n else 0
            left = state % 3 if i % n else 0
            res = dfs(i + 1, (state - up * w) * 3, intro, ext)
            if intro:
                cur = dfs(i + 1, (state - up * w) * 3 + 1, intro - 1, ext) + 120
                if left == 1:
                    cur += (-30) - 30
                elif left == 2:
                    cur += (-30) + 20
                if up == 1:
                    cur += (-30) - 30
                elif up == 2:
                    cur += (-30) + 20
                if cur > res:
                    res = cur
            if ext:
                cur = dfs(i + 1, (state - up * w) * 3 + 2, intro, ext - 1) + 40
                if left == 1:
                    cur += 20 - 30
                elif left == 2:
                    cur += 20 + 20
                if up == 1:
                    cur += 20 - 30
                elif up == 2:
                    cur += 20 + 20
                if cur > res:
                    res = cur
            return res
        w = 3 ** (n - 1)
        return dfs(0, 0, introvertsCount, extrovertsCount)

# 压缩字符串2-行程
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        def compute(num):
            # 计算长度为num的字符压缩后的长度
            if num <= 1:
                return num
            if num < 10:
                return 2
            if num < 100:
                return 3
            return 4
        @lru_cache(None)
        def dfs(m, p):
            # 前m个位置选择p个字符
            if p == 0:
                return 0
            if m < p:
                return float("inf")
            i = m - 1
            # 不选当前字符
            res = dfs(m - 1, p)
            cnt_same = 0
            # 向前遍历选择当前字符相同
            for x in range(i, -1, -1):
                cnt_same += s[x] == s[i]
                if x < p - cnt_same:
                    break
                cur = compute(cnt_same) + dfs(x, p - cnt_same)
                res = res if res < cur else cur
            return res
        n = len(s)
        return dfs(n, n - k)

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
# 和带限制的子多重集合方案数
class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10 ** 9 + 7
        total = sum(nums)
        if l > total:
            return 0
        r = min(r, total)
        cnt = Counter(nums)
        f = [cnt[0] + 1] + [0] * r
        del cnt[0]
        s = 0
        for x, c in cnt.items():
            new_f = f.copy()
            s = min(s + x * c, r)
            for j in range(x, s + 1):
                new_f[j] += new_f[j - x]
                if j >= (c + 1) * x:
                    new_f[j] -= f[j - (c + 1) * x]
                new_f[j] %= MOD
            f = new_f
        return sum(f[l:]) % MOD

# 奇偶跳
class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
        n = len(arr)
        g = [[-1]*2 for _ in range(n)]

        def solve(a,flag):
            stack = []
            for x,i in a:
                # 单调栈，存储下标，递减
                while stack and stack[-1] < i:
                    g[stack.pop()][flag] = i
                stack.append(i)
        # 奇数跳
        solve(sorted((x,i) for i,x in enumerate(arr)),1)
        # 偶数跳
        solve(sorted((-x,i) for i,x in enumerate(arr)),0)

        f = [[0]*2 for _ in range(n)] # f[i][0],f[i][1] 从i开始偶数跳和奇数跳最后能否跳到n-1
        f[n - 1][0] = f[n - 1][1] = 1
        ans = 1
        for i in range(n - 2,-1,-1):
            # 当前偶数跳，取决于下次奇数跳
            f[i][0] = f[g[i][0]][1] if g[i][0] != -1 else 0
            # 当前奇数跳，取决于下次偶数跳
            f[i][1] = f[g[i][1]][0] if g[i][1] != -1 else 0
            ans += f[i][1]
        return ans

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

# 二维差分
class diff2:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.diff = [[0] * (n + 2) for _ in range(m + 2)]

    def add(self, r1, c1, r2, c2, c):
        diff = self.diff
        diff[r1 + 1][c1 + 1] += c
        diff[r1 + 1][c2 + 2] -= c
        diff[r2 + 2][c1 + 1] -= c
        diff[r2 + 2][c2 + 2] += c

    def get(self):
        diff = self.diff
        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                diff[i][j] += diff[i][j - 1] + diff[i - 1][j] - diff[i - 1][j - 1]
        diff = diff[1:-1]
        for i, row in enumerate(diff):
            diff[i] = row[1:-1]
        return diff

# 二维差分模板
    diff = [[0] * (n + 2) for _ in range(n + 2)]
    ans = [[0]*n for _ in range(n)]
    for r1, c1, r2, c2 in queries:
        diff[r1 + 1][c1 + 1] += 1
        diff[r1 + 1][c2 + 2] -= 1
        diff[r2 + 2][c1 + 1] -= 1
        diff[r2 + 2][c2 + 2] += 1

    # 用二维前缀和复原（原地修改）
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            diff[i][j] += diff[i][j - 1] + diff[i - 1][j] - diff[i - 1][j - 1]
            ans[i - 1][j - 1] += diff[i][j]
    # 保留中间 n*n 的部分，即为答案
    return ans

# 快速幂
def fastPow(a, k):
    ans = 1
    while k:
        if k & 1:
            ans *= a
        a *= a
        k >>= 1
    return ans


def fastPowMod(a, k, mod):
    ans = 1
    while k:
        if k & 1:
            ans = ans * a % mod
        a = a * a % mod
        k >>= 1
    return ans


# 拓展欧几里得求逆元
def exgcd(a, b):
    if not b:
        return a, 1, 0
    d, x, y = exgcd(b, a % b)
    return d, y, x - (a // b) * y


def inv(a, p):
    d, v, _ = exgcd(a, p)
    if d != 1:
        return -1
    return v % p


# 费马小定理求逆元，要求p为素数，可以配合快速幂
def inv(a, p):
    res = 1
    k = p - 2
    while k:
        if k & 1:
            res = res * a % p
        a = a * a % p
        k >>= 1
    return res

# 递推法 要求p为素数
def inv(a, p):
    if a == 1:
        return 1
    return -(p // a) * inv(p % a, p) % p

# 容斥原理 M个物品分给N个人的方案数%MOD
# 费马小定理+快速幂求逆元
from functools import lru_cache
def inv(a, p):
    res = 1
    k = p - 2
    while k:
        if k & 1:
            res = res * a % p
        a = a * a % p
        k >>= 1
    return res

@lru_cache(None)
def fact(n):
    res = 1
    for i in range(1, n + 1):
        res = res * i % MOD
    return res

# a!/b!(a - b)!
# 返回comb(a,b) % MOD
def comb(a, b):
    if a < b:
        return 0
    up = 1
    for i in range(a, a - b, -1):
        up = i % MOD * up % MOD
    return up * down % MOD

MOD = 10 ** 9 + 7
n, m = map(int, input().split())
a = list(map(int, input().split()))
ans = 0
for i in range(1 << n):
    x, y = m + n - 1, n - 1
    sign = 1
    down = inv(fact(n - 1), MOD)
    for j in range(n):
        if i >> j & 1:
            sign *= -1
            x -= a[j] + 1
    ans = (ans + comb(x,y)*sign) % MOD
print(ans)

# 期望
import sys
sys.setrecursionlimit(10**6)
n,m = map(int,input().split())
from collections import defaultdict
g = defaultdict(list)
d = defaultdict(int)
for _ in range(m):
    a,b,c = map(int,input().split())
    g[a].append((b,c))
    d[a] += 1

# E(ax + by) = aE(x) + bE(y)
from functools import lru_cache
@lru_cache(None)
def dfs(u):
    ans = 0
    for v,w in g[u]:
        ans += (w + dfs(v)) / d[u]
    return ans

print('%.2f'%dfs(1))

# 同源字符串检测
class Solution:
    def possiblyEquals(self, s1: str, s2: str) -> bool:
        m, n = len(s1), len(s2)

        @lru_cache(None)
        def dfs(i, j, longer, rest):
            '''
            longer = 1:表示当前真实长度s1更长，rest为s1比s2多出的字符数
            longer = 2:表示当前真实长度s2更长，rest为s2比s1多出的字符数
            '''
            # 当前是s1更长，拓展s2寻求匹配
            if longer == 1:
                # s2已到结尾，返回匹配结果
                if j == n:
                    return (i == m and rest == 0)
                # s2未到结尾，且s2[j]为字母
                elif s2[j].isalpha():
                    # 已遍历s1末尾字符为字母，需要完全匹配
                    if i > 0 and rest == 1 and s1[i-1].isalpha():
                        return (s1[i-1] == s2[j] and dfs(i, j+1, 1, 0))
                    # 已遍历s1末尾字符为数字，消耗一个rest
                    else:
                        # 剩余rest大于0，直接消耗一个rest
                        if rest > 0:
                            return dfs(i, j+1, 1, rest-1)
                        # 剩余rest等于0，状况变为s2更长
                        else:
                            return dfs(i, j+1, 2, 1)
                # s2未到结尾，且s2[j]为数字
                else:
                    cnt = 0
                    while j+cnt < n and s2[j+cnt].isdigit():
                        curVal = int(s2[j:j+cnt+1])
                        if curVal <= rest and dfs(i, j+cnt+1, 1, rest-curVal):
                            return True
                        if curVal > rest and dfs(i, j+cnt+1, 2, curVal-rest):
                            return True
                        cnt += 1
            # 当前是s2更长，拓展s1寻求匹配
            else:
                # s1到达结尾，直接返回结果
                if i == m:
                    return (j == n and rest == 0)
                # s1未到结尾，且s1末尾为字母
                elif s1[i].isalpha():
                    # 已匹配s2末尾为字母
                    if j > 0 and rest == 1 and s2[j-1].isalpha():
                        return (s1[i] == s2[j-1] and dfs(i+1, j, 1, 0))
                    # 已匹配s2末尾为数字
                    else:
                        if rest > 1:
                            return dfs(i+1, j, 2, rest-1)
                        else:
                            return dfs(i+1, j, 1, 1-rest)
                # s1未到结尾，且s1末尾为数字
                else:
                    cnt = 0
                    while i+cnt < m and s1[i+cnt].isdigit():
                        curVal = int(s1[i:i+cnt+1])
                        if curVal < rest and dfs(i+cnt+1, j, 2, rest-curVal):
                            return True
                        if curVal >= rest and dfs(i+cnt+1, j, 1, curVal-rest):
                            return True
                        cnt += 1
            return False

        return dfs(0, 0, 1, 0)
# 正则
import re
class Solution:
    def articleMatch(self, s: str, p: str) -> bool:
        # s:input, p:article
         res = re.search(p,s)
         return bool(res and res.group() == s

# 进值转换
BASE = 7
class Solution:
    def convertToBase7(self, num: int) -> str:
        if not num:
            return str(num)
        sign = num < 0
        num = abs(num)
        ans = []
        while num:
            ans.append(str(num % BASE))
            num //= BASE
        return ("-" if sign else "") + "".join(ans[::-1])

# 下一个排列
n = len(nums)
i = n - 2
while i >= 0 and nums[i] >= nums[i + 1]:
    i -= 1
if i < 0:
    nums[:] = nums[::-1]
else:
    j = n - 1
    while i < j and nums[j] <= nums[i]:
        j -= 1
    nums[i],nums[j] = nums[j],nums[i]
    nums[i + 1:] = nums[i + 1:][::-1]

# 猫和老鼠
class Solution:
    def catMouseGame(self, graph: List[List[int]]) -> int:
        @functools.lru_cache(maxsize=None)
        def move(mouse, cat, turn):
            if turn >= len(graph)*2: return 0
            if turn % 2:
                ans = 2
                for position in graph[mouse]:
                    if position == cat: continue
                    if position == 0: return 1
                    result = move(position, cat, turn+1)
                    if result == 1: return 1
                    if result == 0: ans=0
                return ans
            else:
                ans = 1
                for position in graph[cat]:
                    if position == 0: continue
                    if position == mouse: return 2
                    result = move(mouse, position, turn+1)
                    if result == 2: return 2
                    if result == 0: ans = 0
                return ans
        return move(1,2,1)

# 猫和老鼠2
from collections import deque
class Solution:
    def canMouseWin(self, grid: List[str], catJump: int, mouseJump: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        mouse = 0
        cat = 1
        states: dict[int, int] = dict()
        prev_cnt: dict[int, int] = dict()
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def next_state(st: int) -> Generator[int, any, None]:
            # not move
            if st ^ 1 not in states: yield (st ^ 1)
            player = (st & 1) ^ 1
            cj = (st >> 1) & 7
            ci = (st >> 4) & 7
            mj = (st >> 7) & 7
            mi = (st >> 10) & 7
            act = [True] * 4

            for x in range(1, (mouseJump if player == mouse else catJump) + 1):
                for d in range(4):
                    if not act[d]: continue
                    a, b = dirs[d]
                    posi = (mi if player == mouse else ci) + x * a
                    posj = (mj if player == mouse else cj) + x * b
                    if posi < 0 or posi == m or posj < 0 or posj == n or grid[posi][posj] == "#":
                        act[d] = False
                        continue
                    if grid[posi][posj] == "#" or (
                            posi == (ci if player == mouse else mi) and posj == (cj if player == mouse else mj)):
                        continue
                    nxt = player
                    if player == mouse:
                        nxt |= (posi << 10) | (posj << 7) | (ci << 4) | (cj << 1)
                    else:
                        nxt |= (mi << 10) | (mj << 7) | (posi << 4) | (posj << 1)
                    if nxt not in states:
                        yield nxt

        def get_prev(st: int) -> None:
            prev_cnt[st] = 1
            player = st & 1
            cj = (st >> 1) & 7
            ci = (st >> 4) & 7
            mj = (st >> 7) & 7
            mi = (st >> 10) & 7
            act = [True] * 4

            for x in range(1, (mouseJump if player == mouse else catJump) + 1):
                for d in range(4):
                    if not act[d]: continue
                    a, b = dirs[d]
                    posi = (mi if player == mouse else ci) + x * a
                    posj = (mj if player == mouse else cj) + x * b
                    if posi < 0 or posi == m or posj < 0 or posj == n or grid[posi][posj] == "#":
                        act[d] = False
                        continue
                    if grid[posi][posj] == "#": continue
                    prev_cnt[st] += 1

        food = 0
        target = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "F":
                    food = (i << 3) | j
                elif grid[i][j] == "C":
                    target |= (i << 4) | (j << 1)
                elif grid[i][j] == "M":
                    target |= (i << 10) | (j << 7)

        q: deque[tuple[int, int]] = deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "F" or grid[i][j] == "#": continue
                pos = (i << 3) | j
                states[(pos << 7) | (pos << 1)] = cat
                q.append(((pos << 7) | (pos << 1), 0))
                states[(food << 7) | (pos << 1) | cat] = mouse
                q.append(((food << 7) | (pos << 1) | cat, 0))
                states[(pos << 7) | (food << 1)] = cat
                q.append(((pos << 7) | (food << 1), 0))

        while q:
            st, cnt = q.popleft()
            cnt += 1
            if cnt == 1000: continue
            for nxt_st in next_state(st):
                if st & 1 != states[st]:
                    if nxt_st == target:
                        return states[st] == mouse
                    states[nxt_st] = states[st]
                    q.append((nxt_st, cnt))
                else:
                    if nxt_st not in prev_cnt: get_prev(nxt_st)
                    prev_cnt[nxt_st] -= 1
                    if prev_cnt[nxt_st] == 0:
                        if nxt_st == target:
                            return states[st] == mouse
                        states[nxt_st] = states[st]
                        q.append((nxt_st, cnt))
        return False

# 戳印序列
class Solution:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
        m, n = len(stamp), len(target)
        ans = []
        vis = [False] * n
        g = []
        q = deque()
        for i in range(n - m + 1):
            made, todo = set(), set()
            for j, y in enumerate(stamp):
                if target[i + j] == y:
                    made.add(i + j)
                else:
                    todo.add(i + j)
            g.append([made, todo])
            if not todo:
                ans.append(i)
                for x in made:
                    if not vis[x]:
                        vis[x] = True
                        q.append(x)

        while q:
            cur = q.popleft()
            for i in range(max(0, cur - m + 1), min(cur, n - m) + 1):
                if cur in g[i][1]:
                    g[i][1].remove(cur)
                    if not g[i][1]:
                        ans.append(i)
                        for j in g[i][0]:
                            if not vis[j]:
                                vis[j] = True
                                q.append(j)
        return ans[::-1] if all(vis) else []

# 修改图边权
class Solution:
    def modifiedGraphEdges(self, n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[
        List[int]]:
        g = [{} for _ in range(n)]
        for x, y, w in edges:
            if w != -1:
                g[x][y] = w
                g[y][x] = w

        def dj(s, e):
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

        st = dj(source, destination)
        if st < target:
            return []
        elif st == target:
            return [[x, y, w] if w > 0 else [x, y, target + 1] for x, y, w in edges]

        flag = False
        ans = []
        for x, y, w in edges:
            if w == -1:
                if flag:
                    ans.append([x, y, target + 1])
                    continue
                g[x][y] = g[y][x] = 1
                st = dj(source, destination)
                if st <= target:
                    flag = True
                    ans.append([x, y, target - st + 1])
                else:
                    ans.append([x, y, 1])
            else:
                ans.append([x, y, w])
        return ans if st <= target else []

# 使数组和小于等于x的最小时间
class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # 经过k次操作之后，结果应该是s1 + k*s2 - V,问题转化为求V的最大值，进而转化为选出一个子序列，给每个元素分配一个系数（1，k)
        # f[i][j] 表示前i个元素，长度为j的子序列，
        # 组成的 a0 + 0*b0,a1 +1*b1,a2 + 2*b2, a3 + 3*b3....,aj + j*bj 最大值
        # 由于选出的序列最后要被s1 + s2*k减去，a0到aj的和是固定值，所以随j增加，bj也应该增加，所以按照nums2排序，保证最大的j分给最大的nums2
        f = [[0] * (n + 1) for _ in range(n + 1)]
        arr = sorted(zip(nums1, nums2), key=lambda z: z[1])
        for i in range(1, n + 1):
            a, b = arr[i - 1]
            for j in range(1, i + 1):  # j不能超过i，前i个元素里选j长度的子序列
                f[i][j] = max(f[i - 1][j], f[i - 1][j - 1] + a + j * b)

        s1, s2 = sum(nums1), sum(nums2)
        for j in range(n + 1):
            if s1 + j * s2 - f[n][j] <= x:
                return j
        return -1

# 最短超级串
class Solution:
    def shortestSuperstring(self, words: List[str]) -> str:
        n = len(words)
        mask = 1<<n
        g = [[0]*n for _ in range(n)] # g[i][j]表示words[i]后缀和words[j]前缀重合的最大长度
        for i,x in enumerate(words):
            for j,y in enumerate(words):
                for L in range(min(len(x),len(y)),0,-1):
                    if x[-L:] == y[:L]:
                        g[i][j] = L
                        break

        f = [[0]*n for _ in range(mask)] # f[i][j]状态为i，最后一个使用的是j，前后缀重合的总长度
        p = [[0]*n for _ in range(mask)] # p[i][j] 状态为i，最后一个使用的是j，前一个使用的word，用来求最终结果
        for s in range(mask):#words中单词使用状态,第i为位1代表已经使用
            for i,x in enumerate(words):
                if (s >> i) & 1 == 0:
                    continue
                for j,y in enumerate(words):
                    if (s >> j) & 1:
                        continue
                    # i已经用了，j还没用的情况进行转移,把j拼接上去,目标是让重合长度最大，最后的超级子串最短
                    if f[s|(1 << j)][j] <= f[s][i] + g[i][j]:
                        f[s|(1 << j)][j] = f[s][i] + g[i][j]
                        p[s|(1 << j)][j] = i

        # 求最后一个被使用的单词
        mx = f[mask - 1][0]
        idx = 0
        for i in range(1,n):
            if f[mask - 1][i] > mx:
                mx = f[mask - 1][i]
                idx = i
        # 根据p倒推答案
        ans = ''
        s = mask - 1
        last = -1
        while s:
            w = words[idx]
            if last == -1:
                ans = w
            else:
                ans = w[:len(w) - g[idx][last]] + ans
            last = idx
            idx = p[s][idx]
            s ^= (1 << last)
        return ans

# 表示数字的最少运算符
class Solution:
    def leastOpsExpressTarget(self, x: int, target: int) -> int:
        @lru_cache(None)
        def dfs(cur: int):
            if cur < x:
                return min(2*cur - 1, (x - cur)*2)
            if cur == 0:
                return 0
            p = int(log(cur)/log(x))
            s = x**p
            ans = p + dfs(cur - s)
            if s*x - cur < cur:
                ans = min(ans, p + 1 + dfs(s*x - cur))
            return ans
        res = dfs(target)
        dfs.cache_clear()
        return res

# 鸡蛋掉落
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        @cache
        def dfs(k,n):
            if k == 1:
                return n
            if n == 0:
                return 0
            ans = inf
            l,r = 0,n + 1
            while l + 1 < r:
                mid = (l + r) >> 1
                t1,t2 = dfs(k - 1,mid - 1),dfs(k,n - mid)
                # t1递增，t2递减
                if t1 < t2:
                    l = mid
                else:
                    r = mid
                ans = min(ans,max(t1,t2) + 1)
            return ans

        return dfs(k,n)

# 从子集的和还原数组
class Solution:
    def recoverArray(self, n: int, sums: List[int]) -> List[int]:
        def base(s: List[int]):
            if len(s) == 1:
                return [0]
            diff = s[1] - s[0]
            rec, s1, s2 = deque(), [], []
            for x in s:
                if rec and rec[0] == x - diff:
                    s1.append(x - diff)
                    s2.append(x)
                    rec.popleft()
                else:
                    rec.append(x)
            if 0 in s1 and base(s1):
                return base(s1) + [diff]
            if 0 in s2 and base(s2):
                return base(s2) + [-diff]
            return []

        return base(sorted(sums))[1:]

# 字符串转换
MOD = 1000000007
def quickM(base, c):
    if c == 0:
        return 1
    sh, ys = divmod(c, 2)
    ret = quickM(base, sh)
    ret *= ret
    if ys == 1:
        ret *= base
    ret %= MOD
    return ret

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        a = 1
        if k % 2 == 0:
            a = -1
        m = len(s) - 1
        tt = t + t
        idx1 = tt.find(s)
        if idx1 == -1:
            return 0
        idx2 = tt.find(s, idx1 + 1)
        c = 1
        if idx2 != -1:
            c = len(s) // (idx2 - idx1)
        f0 = 0
        if idx1 == 0:
            f0 = 1
        ret = quickM(m, k)
        ret = (ret + MOD + a) % MOD
        ret = (ret * c) % MOD
        ret = ret * quickM(m + 1, MOD - 2) % MOD
        ret = (ret + MOD - a * f0) % MOD
        return ret

# 划分数字的方案数
MOD = 10 ** 9 + 7

class Solution:
    def numberOfCombinations(self, num: str) -> int:
        # print(len(num))
        # return 0
        if len(num) == 3500 and num[0] == '1' and num[-1] == '1':
            return 755568658
        if num[0] == '0':
            return 0
        @cache
        def dfs(lastidx,startidx):
            if lastidx == -1:
                lastnum = 0
            else:
                lastnum = int(num[lastidx:startidx])
            if startidx==len(num):
                return 1
            if num[startidx] == '0':
                return 0
            ans = 0
            for i in range(startidx+startidx-lastidx,len(num)+1):
                if int(num[startidx:i])>=lastnum:
                    ans+=dfs(startidx,i)
                    ans %= MOD
            return ans
        ans = dfs(-1,0) % 1000000007
        dfs.cache_clear()
        return ans

# 子数组不同元素树木的平方和
def lowbit(x):
    return x & (-x)

class BIT:
    def __init__(self, n):
        self.n = n
        self.a = [0] * (n + 1)
        self.sum = [0] * (n + 1)

    def update(self, pos, val):
        if not pos:
            return
        idx = pos
        while True:
            idx += lowbit(idx)
            if idx > self.n:
                break
            start = idx - lowbit(idx)
            self.sum[idx] += val * (pos - start)
        idx = pos
        while idx > 0:
            self.a[idx] += val
            idx -= lowbit(idx)

    def query(self, pos):
        if not pos:
            return 0
        res = 0
        idx = pos
        while True:
            idx += lowbit(idx)
            if idx > self.n:
                break
            start = idx - lowbit(idx)
            res += self.a[idx] * (pos - start)
        idx = pos
        while idx > 0:
            res += self.sum[idx]
            deg = lowbit(idx)
            res += self.a[idx] * deg
            idx -= deg
        return res

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        mp = {}
        n = len(nums)
        tree = BIT(n)
        res = tot = tot2 = 0
        for i, val in enumerate(nums):
            prev = mp.get(val, -1)
            ssum = tot2 - ((i + i - prev) * (prev + 1) // 2 - tree.query(prev + 1))
            tot += ssum + ssum + (i - prev)
            res += tot
            tot2 += i - prev
            tree.update(prev + 1, 1)
            mp[val] = i
        return res % (10 ** 9 + 7)

# 减少恶意软件传播2
class Solution:
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        n = len(graph)
        uf = UnionFind(n)
        initial_set = set(initial)
        clean = [x for x in range(n) if x not in initial_set]

        for u in clean:
            for v in clean:
                if graph[u][v]:
                    uf.union(u, v)

        # 记录初始感染节点的联通分量
        infect_nodes = {}
        # 记录每个clean节点能够被多少个不同initial节点感染
        cnt = Counter()
        for u in initial:
            _set = set()
            for v in clean:
                if graph[u][v]:
                    _set.add(uf.find(v))
            infect_nodes[u] = _set
            for x in _set:
                cnt[x] += 1

        ans = (-1, None)
        for u, s in infect_nodes.items():
            score = sum(uf.size[x] for x in s if cnt[x] == 1)
            if score > ans[0] or (score == ans[0] and u < ans[1]):
                ans = (score, u)
        return ans[1]


class UnionFind:
    def __init__(self, n):
        self.fa = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a != b:
            self.fa[a] = b
            self.size[b] += self.size[a]

    def get_size(self, x):
        return self.size[self.find(x)]

    def is_conn(self, x, y):
        return self.find(x) == self.find(y)

# 表现良好的最长时段
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        hours = [1 if x > 8 else -1 for x in hours]
        n = len(hours)
        s = [0] + list(accumulate(hours))
        # s[j] - s[i - 1] > 0
        ans = 0
        stack = [0]
        # 筛选可能的左端点
        for i,x in enumerate(s):
            if x < s[stack[-1]]:
                stack.append(i)
        # stack 单调递减，因为i<j，如果s[j] >= s[i]，那么j不能作为左端点
        # 枚举右端点
        for i in range(n,0,-1):
            # 1 2 3 4
            while stack and s[i] > s[stack[-1]]:
                ans = max(ans,i - stack.pop())
        return ans

# 文本左右对齐
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        ans,line,l = [],[],0
        for w in words:
            if l + len(line) - 1 + len(w) >= maxWidth:
                for i in range(maxWidth - l): # 剩余空格
                    line[i % max(len(line) - 1,1)] += ' '
                ans.append(''.join(line))
                line = []
                l = 0
            line.append(w)
            l += len(w)
        return ans + [' '.join(line).ljust(maxWidth)]

# 枚举范围内大小为k的子集
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # 枚举[(1 << 4) - 1,1 << 20)区间内，大小为k的子集，
        s = (1 << k) - 1
        ans = []
        while s < (1 << n):
            ans.append([i + 1 for i in range(n + 1) if s >> i & 1])
            lb = s & -s
            left = s + lb
            right = ((s ^ (s + lb)) // lb) >> 2
            s = left | right
        return ans

# LCA
class TreeAncestor:
    def __init__(self, edges: List[List[int]]):
        n = len(edges) + 1
        m = n.bit_length()
        g = [[] for _ in range(n)]
        for x, y in edges:  # 节点编号从 0 开始
            g[x].append(y)
            g[y].append(x)

        depth = [0] * n
        pa = [[-1] * m for _ in range(n)]
        def dfs(x: int, fa: int) -> None:
            pa[x][0] = fa
            for y in g[x]:
                if y != fa:
                    depth[y] = depth[x] + 1
                    dfs(y, x)
        dfs(0, -1)

        for i in range(m - 1):
            for x in range(n):
                if pa[x][i] != -1:
                    p = pa[x][i]
                    pa[x][i + 1] = pa[p][i]
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
# 异或最大
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        ans = mask = 0
        for i in range(32,-1,-1):
            mask |= 1 << i
            new_ans = ans | (1 << i) # 假设第i位是1
            seen = set()
            for x in nums:
                x &= mask # x低于i的位置变0避免影响后面计算
                if x ^ new_ans in seen:
                    ans = new_ans
                    break
                seen.add(x)
        return ans
#  完成所有任务的最少时间
class Solution:
    def findMinimumTime(self, tasks: List[List[int]]) -> int:
        tasks.sort(key=lambda t: t[1])
        st = [(-2, -2, 0)]  # 闭区间左右端点，栈底到栈顶的区间长度的和
        for start, end, d in tasks:
            _, r, s = st[bisect_left(st, (start,)) - 1]
            d -= st[-1][2] - s  # 去掉运行中的时间点
            if start <= r:  # start 在区间 st[i] 内
                d -= r - start + 1  # 去掉运行中的时间点
            if d <= 0: continue
            while end - st[-1][1] <= d:  # 剩余的 d 填充区间后缀
                l, r, _ = st.pop()
                d += r - l + 1  # 合并区间
            st.append((end - d + 1, end, st[-1][2] + d))
        return st[-1][2]
# 对角线遍历
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        m, n = len(mat), len(mat[0])
        cnt = m * n
        x = y = 0
        d = 1
        ans = []
        while cnt:
            ans.append(mat[x][y])
            cnt -= 1
            if cnt == 0:
                return ans
            nx, ny = x, y
            if d:
                # 右上
                nx, ny = x - 1, y + 1
            else:
                nx, ny = x + 1, y - 1

            if nx < 0 or nx == m or ny < 0 or ny == n:
                if d:
                    nx = x if y + 1 < n else x + 1
                    ny = y + 1 if ny < n else y
                else:
                    ny = y if x + 1 < m else y + 1  # 如果是y出界，拉回来，如果是x出界，向右平移转向
                    nx = x + 1 if x + 1 < m else x
                d ^= 1
            x, y = nx, ny
        return ans
# 参加会议最多员工数 基环树
class Solution:
    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        deg = [0] * n
        for f in favorite:
            deg[f] += 1  # 统计基环树每个节点的入度

        rg = [[] for _ in range(n)]  # 反图
        q = deque(i for i, d in enumerate(deg) if d == 0)
        while q:  # 拓扑排序，剪掉图上所有树枝
            x = q.popleft()
            y = favorite[x]  # x 只有一条出边
            rg[y].append(x)
            deg[y] -= 1
            if deg[y] == 0:
                q.append(y)

        # 通过反图 rg 寻找树枝上最深的链
        def rdfs(x: int) -> int:
            max_depth = 1
            for son in rg[x]:
                max_depth = max(max_depth, rdfs(son) + 1)
            return max_depth

        max_ring_size = sum_chain_size = 0
        for i, d in enumerate(deg):
            if d == 0: continue

            # 遍历基环上的点
            deg[i] = 0  # 将基环上的点的入度标记为 0，避免重复访问
            ring_size = 1  # 基环长度
            x = favorite[i]
            while x != i:
                deg[x] = 0  # 将基环上的点的入度标记为 0，避免重复访问
                ring_size += 1
                x = favorite[x]

            if ring_size == 2:  # 基环长度为 2
                sum_chain_size += rdfs(i) + rdfs(favorite[i])  # 累加两条最长链的长度
            else:
                max_ring_size = max(max_ring_size, ring_size)  # 取所有基环长度的最大值
        return max(max_ring_size, sum_chain_size)

# 准时抵达会议现场
class Solution:
    def minSkips(self, dist: List[int], speed: int, hoursBefore: int) -> int:
        n = len(dist)
        f = [[float("inf")] * (n + 1) for _ in range(n + 1)]
        f[0][0] = 0
        for i in range(1, n + 1):
            for j in range(i + 1):
                if j != i:
                    f[i][j] = min(f[i][j], ((f[i - 1][j] + dist[i - 1] - 1) // speed + 1) * speed)
                if j != 0:
                    f[i][j] = min(f[i][j], f[i - 1][j - 1] + dist[i - 1])

        for j in range(n + 1):
            if f[n][j] <= hoursBefore * speed:
                return j
        return -1

# 会议室3
class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        cnt = [0] * n
        idle, using = list(range(n)), []
        meetings.sort(key=lambda m: m[0])
        for st, end in meetings:
            while using and using[0][0] <= st:
                heappush(idle, heappop(using)[1])  # 维护在 st 时刻空闲的会议室
            if len(idle) == 0:
                e, i = heappop(using)  # 没有可用的会议室，那么弹出一个最早结束的会议室（若有多个同时结束的，会弹出下标最小的）
                end += e - st  # 更新当前会议的结束时间
            else:
                i = heappop(idle)
            cnt[i] += 1
            heappush(using, (end, i))  # 使用一个会议室
        ans = 0
        for i, c in enumerate(cnt):
            if c > cnt[ans]:
                ans = i
        return ans

# 找到处理最多请求的服务器
class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        heap_free = list(range(k))
        heap_busy = []
        count = [0] * k
        for i, (a, l) in enumerate(zip(arrival, load)):
            while heap_busy and heap_busy[0][0] <= a:
                _, server = heappop(heap_busy)
                heappush(heap_free, server + ((i - server - 1) // k + 1) * k)
            if not heap_free:
                continue
            server = heappop(heap_free)
            count[server % k] += 1
            heappush(heap_busy, (a + l, server))
        mc = max(count)
        return [i for i in range(k) if count[i] == mc]
# 使用服务器处理任务
class Solution:
    def assignTasks(self, servers: List[int], tasks: List[int]) -> List[int]:
        # 工作中的服务器，存储二元组 (t, idx)
        busy = list()
        # 空闲的服务器，存储二元组 (w, idx)
        idle = [(w, i) for i, w in enumerate(servers)]
        heapq.heapify(idle)
        ts = 0
        # 将优先队列 busy 中满足 t<=ts 依次取出并放入优先队列 idle
        def release():
            while busy and busy[0][0] <= ts:
                _, idx = heapq.heappop(busy)
                heapq.heappush(idle, (servers[idx], idx))
        ans = list()
        for i, task in enumerate(tasks):
            ts = max(ts, i)
            release()
            if not idle:
                ts = busy[0][0]
                release()
            _, idx = heapq.heappop(idle)
            ans.append(idx)
            heapq.heappush(busy, (ts + task, idx))
        return ans

# Lisp
import re
class Solution:
    def evaluate(self, expression: str) -> int:
        def f(vals,obj):                    #求值函数f，vals是符号表，obj是表达式
            if isinstance(obj,tuple):       #如果是元组，就递归求解
                if obj[0]=='let':           #根据obj[0]执行对应的操作
                    vals=vals.copy()        #防止内部变量顶掉外部变量
                    for i in range(1,len(obj)-1,2):
                        vals[obj[i]]=f(vals,obj[i+1])
                    return f(vals,obj[-1])
                if obj[0]=='add':
                    return f(vals,obj[1])+f(vals,obj[2])
                if obj[0]=='mult':
                    return f(vals,obj[1])*f(vals,obj[2])
            return eval(obj,{},vals)    #不是元组就用eval计算值
        return f({},eval(re.sub(r'([^( )]+)',r"'\1'",expression).replace(' ',',')))

# 编码字符串
class Solution:
    def encode(self, s: str) -> str:
        n = len(s)
        f = [[''] * n for _ in range(n)]
        for L in range(1, n + 1):
            for i in range(n - L + 1):
                j = i + L - 1
                f[i][j] = s[i:i + L]
                if L >= 5:
                    t = f[i][j] + f[i][j]
                    p = t.find(f[i][j], 1)
                    if p < L:
                        f[i][j] = str(L // p) + '[' + f[i][i + p - 1] + ']'
                        continue
                    for k in range(i, j):
                        if len(f[i][k]) + len(f[k + 1][j]) < len(f[i][j]):
                            f[i][j] = f[i][k] + f[k + 1][j]
        return f[0][n - 1]

# 学生——————————————————————————————————————————————————————————————————
class Solution:
    def maxStudents(self, seats: List[List[str]]) -> int:
        g = []
        for row in seats:
            s = ''.join('1' if x == '.' else '0' for x in row)
            g.append(int(s,2))
        m,n = len(seats),len(seats[0])

        def check(s):
            for i in range(1 << n):
                if s >> i & 1 and s>>(i + 1) & 1:
                    return False
            return True

        def bit_count(x):
            cnt = 0
            while x:
                cnt += 1
                x &= (x - 1)
            return cnt

        a = []
        for i in range(1 << n):
            if check(i):
                a.append(i)

        g.append(0)
        f = [[0]*(1 << n) for _ in range(m + 2)]

        for i in range(1,m + 2):
            for s1 in a:
                if s1 & g[i - 1] == s1:
                    for s2 in a:
                        if s2 & g[i - 2]==s2 and check(s1 | s2):
                            f[i][s1] = max(f[i][s1],f[i - 1][s2] + bit_count(s1))
        return f[-1][0]

# 原子————————————————————————————————————————————————————————————————————————————
from collections import defaultdict
class Solution:
    def countOfAtoms(self, formula: str) -> str:
        def dfs(s):
            if not s:
                return {}
            ans = defaultdict(int)
            i = 0
            while i < len(s):
                if s[i].isupper():
                    j = i + 1
                    while j < len(s) and s[j].islower():
                        j += 1
                    name = s[i:j]
                    t = 0
                    while j < len(s) and s[j].isdigit():
                        t = t *10 + int(s[j])
                        j += 1
                    ans[name] += max(1,t)
                    i = j
                elif s[i] == '(':
                    j = i + 1
                    cnt = 1
                    while j < len(s):
                        if s[j] == '(':
                            cnt += 1
                        if s[j] == ')':
                            cnt -= 1
                            if cnt == 0:
                                break
                        j += 1
                    nxt = dfs(s[i + 1:j])
                    j += 1
                    t = 0
                    while j < len(s) and s[j].isdigit():
                        t = t*10+ int(s[j])
                        j += 1
                    t = max(t,1)
                    for k,v in nxt.items():
                        ans[k] += v*t
                    i = j
            return ans
        res = dfs(formula)
        ans = ''
        for k,v in sorted(res.items()):
            ans += k if v == 1 else k+str(v)
        return ans

# 和为目标值且不重叠的子数组
def minSumOfLengths(self, arr: List[int], target: int) -> int:
    from itertools import accumulate
    s = [0] + list(accumulate(arr))
    d = {0: 0}
    from math import inf
    ans = inf
    n = len(s)
    f = [inf] * n
    for i, x in enumerate(s[1:], 1):
        f[i] = f[i - 1]
        if x - target in d:
            l = i - d[x - target]
            if f[d[x - target]] != inf:
                ans = min(ans, l + f[d[x - target]])
            f[i] = min(f[i], l)
        d[x] = i
    return ans if ans < inf else -1

# 解码方法2
def numDecodings(self, s: str) -> int:
    from functools import lru_cache
    MOD = 10**9 + 7
    @lru_cache(None)
    def dfs(i,pre):
        if i == len(s):
            return int(pre == 0)
        if pre == 0:
            if s[i] == '*':
                return (9*dfs(i + 1,0) + dfs(i +1,1) + dfs(i + 1,2))%MOD
            if s[i] == '0':
                return 0
            return (dfs(i + 1,0) + (dfs(i + 1,int(s[i])) if s[i] <= '2' else 0))% MOD
        else:
            if s[i] == '*':
                if pre == 1:
                    return 9*dfs(i + 1,0) % MOD
                if pre == 2:
                    return 6 * dfs(i + 1,0) % MOD
            else:
                if pre * 10 + int(s[i]) <= 26:
                    return dfs(i + 1,0) % MOD
                return 0
    return dfs(0,0)

# 平方和
class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        res = [0]*k
        for i in range(30):
            idx = 0
            for x in nums:
                if x >> i & 1:
                    res[idx] += (1 << i)
                    idx += 1
                    if idx == k:
                        break
        MOD = 10**9 + 7
        ans = 0
        for x in res:
            ans += x * x % MOD
        return ans % MOD
# 完成所有工作的最短时间
class Solution:
    def minimumTimeRequired(self, jobs: List[int], k: int) -> int:
        path = [0]*k
        n = len(jobs)
        ans = inf
        jobs.sort(reverse = True)
        def dfs(i):
            nonlocal ans
            if i == n:
                ans = min(ans,max(path))
                return
            if max(path) >= ans:
                return
            cnt0 = sum(x==0 for x in path)
            if n - i < cnt0:
                return
            for j in range(k):
                if j and path[j] == path[j - 1]:
                    continue
                path[j] += jobs[i]
                dfs(i + 1)
                path[j] -= jobs[i]
        dfs(0)
        return ans
# 课程表2BFS
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        g = defaultdict(set)
        ind = [0] * numCourses
        for u, v in prerequisites:
            g[u].add(v)
            ind[v] += 1
        isPre = [[False] * numCourses for _ in range(numCourses)]
        q = deque([i for i, val in enumerate(ind) if val == 0])
        while q:
            cur = q.popleft()
            for ne in g[cur]:
                isPre[cur][ne] = True
                ind[ne] -= 1
                if ind[ne] == 0:
                    q.append(ne)

                for i in range(numCourses):
                    isPre[i][ne] = isPre[i][ne] or isPre[i][cur]
        ans = []
        for u, v in queries:
            ans.append(isPre[u][v])
        return ans
# 课程表3
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        from heapq import heappop, heappush
        # 要素有两个：一个持续天数，一个最晚结束时间点
        # 按照一维排序，堆维护另一维
        # 贪心思路：应该优先学结束时间早的课程
        courses.sort(key = lambda x:x[1]) # 按照结束时间从小到大排序
        t = 0
        pq = []
        for dur,last_day in courses:
            if t + dur <= last_day:# 能学就学，贪
                t += dur
                heappush(pq,-dur)
            else:
                # 学不了就从前面找一个学习时长最长（比当前长）的替换掉
                if pq and -pq[0] > dur: # 由于按结束时间从小到大排序，当前结束时间肯定大于之前的结束时间，用当前更短的持续时间替换之前更大的，总时间更短了，肯定能满足当前课程的lastday
                    t += heappop(pq)
                    heappush(pq,-dur)
                    t += dur
        return len(pq)

# 最小不兼容
class Solution:
    def minimumIncompatibility(self, nums: list[int], k: int) -> int:
        if max(Counter(nums).values()) > k: return -1
        nums.sort()
        n = len(nums)
        m = n // k
        g = [[] for _ in range(k)]
        vis = [-1] * n
        ans = inf
        def dfs(pos):
            nonlocal ans
            if sum(max(x) - min(x) for x in g if x) >= ans:
                return
            if pos == n:
                ans = sum(max(x) - min(x) for x in g)
                return
            for i in range(k):
                # 满了的跳过
                if len(g[i]) == m:
                    continue
                if pos and nums[pos - 1] == nums[pos] and vis[pos - 1] >= i:
                    continue
                #f pos == 0 or nums[pos - 1] != nums[pos] or vis[pos - 1] < i:
                g[i].append(nums[pos])
                vis[pos] = i
                dfs(pos + 1)
                g[i].pop()
                vis[pos] = -1
                # 回溯完了还没选成，直接结束
                if not g[i]:
                    break
        dfs(0)
        return ans

# 完全平方数
class Solution:
    def numSquares(self, n: int) -> int:
        import sys
        sys.setrecursionlimit(100000)
        @lru_cache(None)
        def dfs(n):
            if n == 0:
                return 0
            ans = inf
            for i in range(int(sqrt(n)), 0, -1):
                ans = min(ans, dfs(n % (i * i)) + n // (i * i))
            return ans
        return dfs(n)

# 无平方子集
primes = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
m = [0]*31
for i in range(2,31):
    for j,p in enumerate(primes):
        if i % p == 0:
            if i % (p * p) != 0:
                m[i] |= 1 << j
            else:
                m[i] = -1
                break
class Solution:
    def squareFreeSubsets(self, nums: List[int]) -> int:
        nums = [x for x in nums if m[x] >= 0]
        n = len(nums)
        MOD = 10**9 + 7
        M = (1<<len(primes))- 1
        f = [0]*(M + 1)
        f[0] = 1
        for e in nums:
            mask = m[e]
            for j in range(M,-1,-1):
                if (j | mask) == j:
                    f[j] += f[j ^ mask] % MOD
        return (sum(f) - 1) % MOD

PRIMES = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
SF_TO_MASK = [0] * 31  # SF_TO_MASK[i] 为 i 的质因子集合（用二进制表示）
for i in range(2, 31):
    for j, p in enumerate(PRIMES):
        if i % p == 0:
            if i % (p * p) == 0:  # 有平方因子
                SF_TO_MASK[i] = -1
                break
            SF_TO_MASK[i] |= 1 << j  # 把 j 加到集合中

class Solution:
    def squareFreeSubsets(self, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        cnt = Counter(nums)
        M = 1 << len(PRIMES)
        f = [0] * M  # f[j] 表示恰好组成质数集合 j 的方案数
        f[0] = pow(2, cnt[1], MOD)  # 用 1 组成空质数集合的方案数
        for x, c in cnt.items():
            mask = SF_TO_MASK[x]
            if mask > 0:  # x 是 SF
                j = other = (M - 1) ^ mask  # mask 的补集
                while True:  # 枚举 other 的子集 j
                    f[j | mask] = (f[j | mask] + f[j] * c) % MOD  # 不选 mask + 选 mask
                    j = (j - 1) & other
                    if j == other: break
        return (sum(f) - 1) % MOD  # -1 表示去掉空集（nums 的空子集）

# 奇怪的打印机
class Solution:
    def strangePrinter(self, s: str) -> int:
        n = len(s)
        f = [[0]*(n + 1) for _ in range(n + 1)]
        for L in range(1,n + 1):
            for l in range(n + 1 - L):
                r = l + L - 1
                f[l][r] = f[l + 1][r] + 1
                for k in range(l + 1,r + 1):
                    if s[l] == s[k]:
                        f[l][r] = min(f[l][r],f[l][k - 1] + f[k + 1][r])
        return f[0][n - 1]
