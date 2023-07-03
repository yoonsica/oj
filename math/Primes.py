"""
质数筛
"""
from math import sqrt


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
