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


print(KMP().strStr('abcdef', 'de'))