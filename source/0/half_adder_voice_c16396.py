# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/sound2jaw/half_adder_voice.py
import numpy as np
import math

# importing Qiskit
import qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer
import os
import sys

# import basic plot tools
from qiskit.visualization import plot_histogram


data= sys.argv[1]
print(data)


qn = 3
cn = 3
creg = QuantumRegister(qn, "count")
areg = QuantumRegister(cn, "ancilla")
clreg = ClassicalRegister(qn, "calssical")
qc = QuantumCircuit(creg, areg, clreg)

for i in range(len(data)):
    if data[i] == "1":
        qc.x(i)
        print("inside")
    else:
        pass
qc.h(qn)
qc.ccx(qn,0,qn+1)
qc.cx(qn,0)
qc.ccx(qn+1,1,qn+2)
qc.cx(qn+1,1)
qc.cx(qn+2,2)
qc.swap(0,2)
qc.measure(creg, clreg)


emulator = Aer.get_backend('qasm_simulator')
job = execute(qc, emulator, shots=1 )
hist = job.result().get_counts()
print(hist)
# save results in csv file; done !