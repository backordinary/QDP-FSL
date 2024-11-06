# https://github.com/Ben-Foxman/feynman_path/blob/5a9c6b5ee75d59e201cc1e5dbee3e068bb5379cf/bb/RouterTree.py
from qiskit import *


class RouterTree(object):
    def __init__(self, size: int) -> None:
        assert size > 0
        self.root = Router(0)
        self.addLeaves(self.root, size)
        self.array = self.createRouterArray(self.root)

    def addLeaves(self, r, size):
        if size == 1:
            return

        root = r
        left = Router(int(bin(root.number)[2:], 2) * 2 + 1)
        right = Router(int(bin(root.number)[2:], 2) * 2 + 2)
        left.parent = root
        right.parent = root
        root.leftChild = left
        root.rightChild = right
        self.addLeaves(root.leftChild, size - 1)
        self.addLeaves(root.rightChild, size - 1)

    def createRouterArray(self, root):
        routers = []
        if root.leftChild:
            routers += self.createRouterArray(root.leftChild)
        routers.append(root)
        if root.rightChild:
            routers += self.createRouterArray(root.rightChild)
        return routers


class Router(object):
    def __init__(
        self, number: int, leftChild=None, rightChild=None, parent=None
    ) -> None:
        self.number = number
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.parent = parent
        self.register = QuantumRegister(3, f"router{bin(number)[2:]}")

    def isLeaf(self):
        return self.leftChild == None and self.rightChild == None

    def isRoot(self):
        return self.parent == None


if __name__ == "main":
    pass