from typing import List

# 对角线遍历

class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        m, n = len(mat), len(mat[0])
        cnt = m * n
        x = y = 0
        d = 1
        ans = []
        while cnt:
            ans.append(mat[x][y])
            cnt -= 1
            if cnt == 0:
                return ans
            nx, ny = x, y
            if d:
                # 右上
                nx, ny = x - 1, y + 1
            else:
                nx, ny = x + 1, y - 1

            if nx < 0 or nx == m or ny < 0 or ny == n:
                if d:
                    nx = x if y + 1 < n else x + 1
                    ny = y + 1 if ny < n else y
                else:
                    ny = y if x + 1 < m else y + 1  # 如果是y出界，拉回来，如果是x出界，向右平移转向
                    nx = x + 1 if x + 1 < m else x
                d ^= 1
            x, y = nx, ny
        return ans
