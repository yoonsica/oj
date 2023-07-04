class Trie:

    def __init__(self):
        self.child = {}
        self.isEnd = False

    def insert(self, word: str) -> None:
        root = self
        for c in word:
            if c not in root.child:
                root.child[c] = Trie()
            root = root.child[c]
        root.isEnd = True

    def search(self, word: str) -> bool:
        root = self
        for c in word:
            if c not in root.child:
                return False
            root = root.child[c]
        return root.isEnd

    def startsWith(self, prefix: str) -> bool:
        root = self
        for c in prefix:
            if c not in root.child:
                return False
            root = root.child[c]
        return True
