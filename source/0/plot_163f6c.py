# https://github.com/islegmar/quantumChangePoint/blob/4edf11710cd070eb66648b708ef4d4bacf0f454f/code/plot.py
# -*- coding: utf-8 -*-
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from numpy import pi
import matplotlib.pyplot as plt

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.x(qreg_q[2]).c_if(creg_c, 1)
circuit.measure(qreg_q[2], creg_c[2])


#circuit.draw(output='mpl')
#plt.show()

simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=1024)
result = job.result()
counts = result.get_counts(circuit)
print (counts)
