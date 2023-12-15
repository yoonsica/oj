# floyd注意dis和g分开
def floyd(d, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])


# floyd求方案
# 求i-j之间的路径，不包含i,j
# def dfs(i, j):
#     if not pos[i][j]:
#         return []  # i,j不通
#     k = pos[i][j]
#     return dfs(i, k) + [k] + dfs(k, j)