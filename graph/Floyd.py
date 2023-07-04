# d[0][0] = 0
def floyd(d, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][k], d[k][j])
