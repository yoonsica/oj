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
