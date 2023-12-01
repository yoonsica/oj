# ll exgcd(ll a, ll b, ll &x, ll &y)// 拓欧
# {
#     if (b == 0)
#     {
#         x = 1;
#         y = 0;
#         return a;
#     }
#     ll d = exgcd(b, a % b, y, x);
#     y -= (a / b) * x;
#     return d;
# }
# ll inv(ll a, ll p)
# {
#     ll x, y;
#     if (exgcd(a, p, x, y) != 1) // 无解的情形
#         return -1;
#     return (x % p + p) % p;
# }

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


# 费马小定理求逆元，要求p为素数
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
