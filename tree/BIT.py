class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.m = len(matrix)
        self.n = len(matrix[0])
        self.nums = []
        for row in matrix:
            self.nums += row
        self.t = BIT(self.nums)

    def update(self, row: int, col: int, val: int) -> None:
        idx = self.n * row + col
        self.t.add(idx + 1, val - self.nums[idx])
        self.nums[idx] = val

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        ans = 0
        for r in range(row1, row2 + 1):
            ans += self.t.query(self.n * r + col2 + 1) - self.t.query(self.n * r + col1)
        return ans


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# obj.update(row,col,val)
# param_2 = obj.sumRegion(row1,col1,row2,col2)

class BIT:
    def __init__(self, nums):
        self.tree = [0] * (len(nums) + 1)
        for i, x in enumerate(nums, 1):
            self.add(i, x)

    def add(self, i, v):
        while i < len(self.tree):
            self.tree[i] += v
            i += i & -i

    def query(self, i):
        ans = 0
        while i:
            ans += self.tree[i]
            i &= i - 1
        return ans