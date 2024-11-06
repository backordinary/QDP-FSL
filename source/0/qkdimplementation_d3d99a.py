# https://github.com/vexandmore/QKDproject/blob/45ed1e53e3ac52d8bb3878afd137a4e25359c16a/src/main/qkdImplementation.py
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:14:41 2021

@author: Marc
"""
from numpy.random import *
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
import sys


def measureMessage(sendCircuits, bases, backend):
        measurements = []
        for i in range(len(bases)):
            if (bases[i] == 1):
                sendCircuits[i].h(0)
                sendCircuits[i].measure(0,0)
                sendCircuits[i].h(0)
            else:
                sendCircuits[i].measure(0,0)
            result = execute(sendCircuits[i], backend, shots=1, memory=True).result()
            receiver_bit = int(result.get_memory()[0])
            measurements.append(receiver_bit)
        return measurements


"""So 1 represents either |1> or |->, 0 represents |0> or |+> for bits
For the bases, 0 represents Z basis and 1 represents X basis"""
class KeySender:
    def __randbits__(num):
        gen = default_rng()
        return gen.integers(0,1,num, endpoint=True)

    def __init__ (self, num, bits=[], bases=[]):
        if bits == []:
            bits = KeySender.__randbits__(num)
        if bases == []:
            bases = KeySender.__randbits__(num)
        
        self.num = num
        self.sendBits = bits
        self.sendBases = bases
        self.matchingIndices = []#indicies where sender and this measured in the same axis
        self.__privateKeyData = []
        self.compareKeyData = []
        
    def makeSendCircuits(keyData):
        circuits = []
        for i in range(len(keyData.sendBits)):
            circuit = QuantumCircuit(1,1)
            if (keyData.sendBits[i] == 1):
                circuit.x(0)
            if (keyData.sendBases[i] == 1):
                circuit.h(0)
            circuits.append(circuit)
        return circuits


class Eavesdropper:
    def __randbits__(num):
        gen = default_rng()
        return gen.integers(0,1,num, endpoint=True)
    
    def __init__(self, num, bases = []):
        if bases == []:
            bases = Eve.__randbits__(num)
        self.num = num            
        self.receiveBases = bases
        self.receivedBits = []
    
    def measure(self, circuits, backend):
        self.receivedBits = measureMessage(circuits, self.receiveBases, backend)
        # for i in range(len(self.receiveBases)):
        #     if (self.receiveBases[i] == 1):
        #         #if measuring in X basis, use an H before and after
        #         #so that if Bob measures in X as well, should measure the same
        #         circuits[i].h(0)
        #         circuits[i].measure(0,0)
        #         circuits[i].h(0)
        #     else:
        #         circuits[i].measure(0,0)
        #     result = execute(circuits[i], backend, shots=1, memory=True).result()
        #     receiver_bit = int(result.get_memory()[0])
        #     self.receivedBits.append(receiver_bit)
        
    def printMeasured(self):
        for i in self.receivedBits:
            print(i, end='')
        



class KeyReceiver:
    def __randbits__(num):
        gen = default_rng()
        return gen.integers(0,1,num, endpoint=True)

    def __init__(self, num, bases = []):
        if bases == []:
            bases = KeyReceiver.__randbits__(num)
        
        self.num = num
        self.receivedBits = []
        self.receiveBases = bases
        self.matchingIndices = []
        self.__privateKeyData = []
        self.compareKeyData = []

    def measure(self, circuits, backend):
        self.receivedBits = measureMessage(circuits, self.receiveBases, backend)

    def printMeasured(self):
        for i in self.receivedBits:
            print(i, end='')


def ReceiveData(sender, receiver, eve=None, backend = Aer.get_backend('qasm_simulator')):
    if eve is None:
        ReceiveDataNormal(sender, receiver, backend)
    else:
        ReceiveDataEavesdropper(sender, receiver, eve, backend)

def ReceiveDataEavesdropper(sender, receiver, eve, backend):
    #Send the bits across the channel
    sendCircuits = sender.makeSendCircuits()
    eve.measure(sendCircuits, backend)
    receiver.measure(sendCircuits, backend)

def ReceiveDataNormal(sender, receiver, backend):
    #Send the bits across the channel
    sendCircuits = sender.makeSendCircuits()
    receiver.measure(sendCircuits, backend)

"""Turns a string containing 1s and 0s into a list of 1s and 0s"""
def listFromString(str):
    list = []
    for i in str:
        list.append(int(i))
    return list


def main():
    while True:
        inArgs = input().split()
        keyLength = len(inArgs[0])
        aliceBits = listFromString(inArgs[0])
        aliceBases = listFromString(inArgs[1])
        if len(inArgs) > 3:
            #have an eavesdropper
            Eve = Eavesdropper(keyLength, listFromString(inArgs[2]))
            bobBases = listFromString(inArgs[3])
        else:
            #No eavesdropper
            bobBases = listFromString(inArgs[2])
            Eve = None
        
        Alice = KeySender(keyLength, aliceBits, aliceBases)
        Bob = KeyReceiver(keyLength, bobBases)        
        
        ReceiveData(Alice, Bob, Eve)
        Bob.printMeasured()
        print(' ', end='')
        if Eve is not None:
            Eve.printMeasured()
        print('')
main()












