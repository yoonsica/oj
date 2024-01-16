# 生成回文数
# 预处理出1e9以内的回文数
pal = []
base = 1
while base <= 10000: # 99999999
    # 生成奇数长度回文数 12321
    for i in range(base, base * 10):
        x = i
        t = i // 10
        while t:
            x = x * 10 + t % 10
            t //= 10
        pal.append(x)
    # 生成偶数长度回文数 123321
    if base <= 1000:
        for i in range(base, base * 10):
            x = t = i
            while t:
                x = x * 10 + t % 10
                t //= 10
            pal.append(x)
    base *= 10
