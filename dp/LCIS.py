from math import inf
# 最长公共上升子序列
# a,b是长分别为m,n的两个数组，找出a,b中公共上升子序列的长度
# dp定义：f[i][j]为a的前i个数、b的前j个数中以b[j]结尾的（可以包含a[j],也可以不包含）最长上升子序列的长度
# 最终答案为max(f[m][j] for j in range(1,n+ 1))
# 朴素解法O（n3）
a = [-inf] + list(map(int,input().split()))
b = [-inf] + list(map(int,input().split()))
m,n = len(a) - 1,len(b) - 1
f = [[0]*(n + 1) for _ in range(m + 1)]
for i in range(1,m + 1):
    for j in range(1,n + 1):
        # 如果不选a[i]
        f[i][j] = f[i - 1][j]
        # 如果选a[i]，那么a[i] == b[j]
        if a[i] == b[j]:
            # 转移到b[k],要求b[k] < b[j]
            for k in range(j): # 注意k从0开始
                if b[k] < b[j]:
                    f[i][j] = max(f[i][j],1 + f[i - 1][k])
print(max(f[m][j] for j in range(1,n + 1)))

# 优化版本O(n2)
# f[i][j] 只和f[i-1]有关
a = [-inf] + list(map(int,input().split()))
b = [-inf] + list(map(int,input().split()))
m,n = len(a) - 1,len(b) - 1
f = [[0]*(n + 1) for _ in range(m + 1)]
for i in range(1,m + 1):
    maxv = 0
    for j in range(1,n + 1):
        # 如果不选a[i]
        f[i][j] = f[i - 1][j]
        # 如果选a[i]，那么a[i] == b[j]
        if a[i] == b[j]:
            f[i][j] = max(f[i][j],1 + maxv)
        if a[i] > b[j]: # a[i]可以作为b[j]之后的b[k]匹配
            maxv = max(maxv, f[i - 1][j])
print(max(f[m][j] for j in range(1,n + 1)))