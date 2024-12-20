# https://github.com/alvinli04/Quantum-Steganography/blob/c6d0ef298bc7da5a71be18df3d785887d3456d92/neqr.py
import qiskit

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import Aer
from qiskit import IBMQ
from qiskit.compiler import transpile
from time import perf_counter
from qiskit.tools.visualization import plot_histogram

import numpy as np
import math

'''
params
---------------
picture: square 2d array of integers representing grayscale values
assume the length (n) is some power of 2

return
---------------
a flattened representation of picture using bitstrings (boolean arrays)
'''
def convert_to_bits (picture):
    n = len(picture)
    ret = []
    for i in range(n):
        for j in range(n):
            value = picture[i][j]
            bitstring = bin(value)[2:]
            ret.append([0 for i in range(8 - len(bitstring))] + [1 if c=='1' else 0 for c in bitstring])
    return ret


'''
params
----------------
bitStr: a representation of an image using bitstrings to represent grayscale values

return
----------------
A quantum circuit containing the NEQR representation of the image
'''
def neqr(bitStr, quantumImage, idx, intensity): 
    newBitStr = bitStr

    numOfQubits = quantumImage.num_qubits

    quantumImage.draw()
    print("Number of Qubits:", numOfQubits)

    # -----------------------------------
    # Drawing the Quantum Circuit
    # -----------------------------------
    lengthIntensity = intensity.size
    lengthIdx = idx.size

    quantumImage.i([intensity[lengthIntensity-1-i] for i in range(lengthIntensity)])
    quantumImage.h([idx[lengthIdx-1-i] for i in range(lengthIdx)])

    numOfPixels = len(newBitStr)

    for i in range(numOfPixels):
        bin_ind = bin(i)[2:]
        bin_ind = (lengthIdx - len(bin_ind)) * '0' + bin_ind
        bin_ind = bin_ind[::-1]

        # X-gate (enabling zero-controlled nature)
        for j in range(len(bin_ind)):
            if bin_ind[j] == '0':
                quantumImage.x(idx[j])

        # Configuring CCNOT (ccx) gates with control and target qubits
        for j in range(len(newBitStr[i])):
            if newBitStr[i][j] == 1:
                quantumImage.mcx(idx, intensity[lengthIntensity-1-j])
        
        # X-gate (enabling zero-controlled nature)
        for j in range(len(bin_ind)):
            if bin_ind[j] == '0':
                quantumImage.x(idx[j])

        quantumImage.barrier()


