# https://github.com/Robinbux/AI-Projects/blob/edde9e02ad21263ad21ad0d3bff64e78c556d587/QKNN/circuit_creator.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates.swap import SwapGate
from numpy import pi

def swap_test():
    qreg_q = QuantumRegister(3, 'qr')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.h(qreg_q[0])
    circuit.cswap(qreg_q[0], qreg_q[1], qreg_q[2])
    circuit.h(qreg_q[0])
    circuit.measure(qreg_q[0], creg_c[0])