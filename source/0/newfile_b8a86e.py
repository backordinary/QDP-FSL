# https://github.com/fs1132429/QueueMeet/blob/5f6fadd473413b23e520ded4c810331823a0a67a/newFile.py
from qiskit import QuantumCircuit, Aer, transpile
from QKDFunctions import *

if __name__ == "__main__":
    # qc = QuantumCircuit(3)
    # qc.x(0)
    # qc.draw()

    key_list = gen_key(100)
    print(''.join(map(str, key_list)))
