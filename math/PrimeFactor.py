# 分解质因数
from collections import defaultdict

nums = []
ans = defaultdict(list)
for i, x in enumerate(nums):
    d = 2
    while d * d <= x:
        if x % d == 0:
            #f(d,i) d是质因数
            ans[i].append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        ans[i].append(x) # f(x,i) x也是质因数
