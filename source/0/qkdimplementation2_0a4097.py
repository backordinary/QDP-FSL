# https://github.com/vexandmore/QKDproject/blob/31fb08e3ca20947dbc8b4aa2855d00d1b02d91e1/qkdImplementation2.py
# -*- coding: utf-8 -*-
"""
This script provides the Alice with her circuits (based on the number of bits
, the bits, and the bases.)
Input format: number of bits bits bases
[Bits and bases are passed to stdin as bistrings, number of bits is passed as
 an integer]
Circuits are returned as a json-wrapped list of qasm.



And also acts as a way to measure the circuits. (with measurement bases and 
                                                 circuits)
Input format: bases circuits
pass in bases as bitstring (either 0 or 1 or 2 to indicate not measuring), 
followed by a space, followed by json consisting in a list of qasm strings.
Measurements are returned as a bitstring (0, 1, or 2), followed by a space, 
followed by the new qasm circuits
@author: Marc
"""

from numpy.random import *
import json
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
import sys
import time

"""Measures message in bases."""
def measureMessage(sendCircuits, bases, backend):
        measurements = []
        for i in range(len(bases)):
            if (bases[i] == 2):
                measurements.append(2)
            else:
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


def listFromString(str):
    list = []
    for i in str:
        list.append(int(i))
    return list


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

def parseQasm(qasm):
    #return QuantumCircuit.from_qasm_str(qasm)
    list = qasm.split("\n")
    if list.pop(0) != "OPENQASM 2.0;":
        raise Exception('bad qasm string on line 1')
    if list.pop(0) != "include \"qelib1.inc\";":
        raise Exception('bad qasm string on line 2')
    #check the number of quantum and classical bits and create circuit
    qreg = int(list.pop(0)[-3])
    creg = int(list.pop(0)[-3])
    circuit = QuantumCircuit(qreg, creg)
    for i in list:
        sub = i.split(' ')
        if sub[0] == "h":
            circuit.h(int(sub[1][-3]))
        elif sub[0] == "x":
            circuit.x(int(sub[1][-3]))
        elif sub[0] == "measure":
            circuit.measure(int(sub[1][-2]), int(sub[3][-3]))
        elif sub[0] == '':
            pass
        else:
            raise Exception('unknown token ' + sub[0] + ' in qasm')
    return circuit
    

def main():
    backend = Aer.get_backend('qasm_simulator')
    while True:
        lineIn = input()
        #send Alice her circuits
        if lineIn[-1] == '0' or lineIn[-1] == '1':
            inArgs = lineIn.split()
            numBits = int(inArgs[0])
            bits = listFromString(inArgs[1])
            bases = listFromString(inArgs[2])
            
            alice = KeySender(numBits, bits, bases)
            circuits = alice.makeSendCircuits()
            qasmCircuits = []
            for i in circuits:
                qasmCircuits.append(i.qasm())
            print(json.dumps(qasmCircuits))
        
        #measure circuits for Bob (or Eve)
        else :  
            inArgs = lineIn.split(' ', 1)
            bases = listFromString(inArgs[0])
            qasmCircuits = json.loads(inArgs[1])
            circuits = []
            for i in qasmCircuits:
                circuits.append(parseQasm(i))
                #circuits.append(QuantumCircuit.from_qasm_str(i))
            measurements = measureMessage(circuits, bases, backend)
            
            #provide results
            for i in measurements:
                print(i, end='')
            print (' ', end='')#Space
            #provide new circuits
            newQasmCircuits = []
            for i in range(len(circuits)):
                newQasmCircuits.append(circuits[i].qasm())
            print(json.dumps(newQasmCircuits))
main()