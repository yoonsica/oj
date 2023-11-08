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

class KMP:
    def strStr(self, haystack: str, needle: str) -> int:
        # 新模版
        def get_next(s):
            j,n = -1,len(s)
            next = [-1]*n
            # 求next数组过程和模式串与匹配串匹配过程应该是一致的,i一直往前，j往回跳或者+1
            for i in range(1,n):
                while j >= 0 and s[i] != s[j + 1]:
                    j = next[j]
                if s[i] == s[j + 1]:
                    j += 1
                next[i] = j
            return next
        next = get_next(needle)
        n = len(needle)
        j,m = -1,len(haystack)
        for i in range(m):
            while j >= 0 and haystack[i] != needle[j + 1]:
                j = next[j]
            if haystack[i] == needle[j + 1]:
                j += 1
            if j == n - 1:
                return i - n + 1
        return -1