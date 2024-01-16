# 二维差分 下标从0开始
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