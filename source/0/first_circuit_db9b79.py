# https://github.com/NeoTRAN001/QuamtumComputing/blob/262cd6713014957d46f50183306143c9c18cfcde/first_circuit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.t(qreg_q[0])
circuit.t(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
