# https://github.com/d-hsi/qiskit_hackathon_sample/blob/8b13a3c842aaa49cf5722a76ef8f5f17e55c1f38/qiskit_hackathon_test.py
import qiskit

from qiskit import IBMQ
IBMQ.save_account ('key')

qiskit.__qiskit_version__



# qiskit
# python 
# sha256
# WeGo_Delta

import hashlib

y = '23dc4da786eff8147c4e72b9807785afee48bb'
password = ["1004"]
#binarizes
def main():
    thash = "1004"
    thashd = thash.encode()
    hash = hashlib.sha256( thashd )
    hexa = hash.hexdigest()
    return hexa 
	
mainHs = main()
' '.join(format(ord(x), 'b') for x in mainHs)


binary = ' '.join(format(ord(x), 'b') for x in mainHs)

split_threads_string = binary.split(' ')

adata = split_threads_string[0]
xdata = adata.split('000')
print(xdata[-1])

# Grover Algo.
# def solveIt():
#     for x in xdata:
#         if x == y:
#             return 1

# print(solveIt())



# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# # importing Qiskit
# from qiskit import BasicAer, IBMQ
# from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, transpile #compile
# from qiskit.tools.visualization import plot_histogram