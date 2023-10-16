# 并查集（带联通分量大小）
class UnionFind:
    def __init__(self,n):
        self.fa = list(range(n))
        self.size = [1]*n
    
    def find(self,x):
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self,x,y):
        a,b = self.find(x),self.find(y)
        if a != b:
            self.fa[a] = b
            self.size[b] += self.size[a]
    
    def get_size(self,x):
        return self.size[self.find(x)]