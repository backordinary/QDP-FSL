# https://github.com/Jh0mpis/GrafosAleatoriosUsandoComputacionCuantica/blob/a73e7261cdb751426104bcff38c2f5a207e7e4de/util.py
import random as rand
from dataStructures import BinaryTree, Queue, DoubleNode
import matplotlib.pyplot as mp
from qiskit import QuantumCircuit

def ROT(n,word):
    newWord=""
    lowerAbecedary = "abcdefghijklmnopqrstuvwxyz"
    upperAbecedary = lowerAbecedary.upper()
    for i in word:
        if i == ' ':
            newWord += i
        else:
            if(i==i.upper()):
                index = upperAbecedary.index(i)
                newWord += upperAbecedary[(index+n)%len(upperAbecedary)]
            else:
                index = lowerAbecedary.index(i)
                newWord += lowerAbecedary[(index+n)%len(upperAbecedary)]
    return newWord

def getRandomClasicBinaryTree(height,intMin=1,intMax=10):
    tree = BinaryTree()
    tree.addKey(0)
    queue = Queue()
    queue.enqueue(tree.root)
    for i in range(height):
        for j in range(2**(i)):
            current = queue.dequeue()
            current.left = DoubleNode(rand.randint(1,25))
            current.right = DoubleNode(rand.randint(1,25))
            queue.enqueue(current.left)
            queue.enqueue(current.right)
    return tree

def getNotRandomClasicBinaryTree(height,intLeft,intRight):
    tree = BinaryTree()
    tree.addKey(0)
    queue = Queue()
    queue.enqueue(tree.root)
    for i in range(height):
        for j in range(2**(i)):
            current = queue.dequeue()
            current.left = DoubleNode(intLeft)
            current.right = DoubleNode(intRight)
            queue.enqueue(current.left)
            queue.enqueue(current.right)
    return tree

def ROTTree(word,tree):
    newWord = ""
    path = []
    current = tree.root
    for i in word:
        current = rand.choice([current.left,current.right])
        path.append(current.data)
        newWord += ROT(current.data,i)
    return newWord,path

def binaryAbecedary():
    abecedaryBinary =[]
    for i in range(26):
        string = format(i,"b")
        abecedaryBinary.append("0"*(5-len(string))+string)
    return abecedaryBinary

def stringToBinary(word):
    abecedary = "abcdefghijklmnopqrstuvwxyz"
    abecedaryBinary = binaryAbecedary()
    newWord = []
    for i in word:
        newWord.append(abecedaryBinary[abecedary.index(i)])
    return newWord

def binaryToString(binary):
    abecedaryBinary = binaryAbecedary()
    abecedary = "abcdefghijklmnopqrstuvwxyz"
    word = ""
    for i in binary:
        word += abecedary[abecedaryBinary.index(i)]
    return word

def prepareCircuit(binary):
    circuit = QuantumCircuit(5,5)
    for i in range(len(binary)):
        if(binary[i]=="1"):
            circuit.x(i)
    circuit.barrier()
    circuit.h(range(5))
    circuit.measure(range(5),range(5))
    return circuit

def binToInt(string):
    num = 0
    for i in range(len(string)):
        num += (2**i)*int(string[len(string)-i-1])
    return num