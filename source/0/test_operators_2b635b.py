# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/Mat3420/test_operators.py
import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import Operator
from array_to_latex import array_to_latex


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


circ = qk.QuantumCircuit(3)

circ.h(2)                                                                       # applying the HADAMARD gate to the third qubit
print('Hadamard')
print(Operator(circ).data)

circ.ccx(0, 1, 2)                                                               # applying the CC NOT (toffoli) gate to the third qubit with 1st and 2nd as reference
print('Hadamard * CCX')
print(Operator(circ).data)

circ.h(2)

print('Hadamard * CCX * Hadamard')
print(Operator(circ).data)

circ.sdg(2)                                                                     # applying the CONJUGATE PHASE GATE (s^-1) to the third cubit

print('Hadamard * CCX * Hadamard * Phase^-1')
print(Operator(circ).data)

circ.ccx(0, 1, 2)

print('Hadamard * CCX * Hadamard * Phase^-1 * CCX')
print(Operator(circ).data)

circ.s(2)                                                                       # applying the PHASE GATE to the third qubit

print('Hadamard * CCX * Hadamard * Phase^-1 * CCX * Phase')
print(Operator(circ).data)

circ.ccx(0, 1, 2)

print('Hadamard * CCX * Hadamard * Phase^-1 * CCX * Phase * CCX')
print(Operator(circ).data)


print(circ)
output = array_to_latex(Operator(circ).data)
print(output)

#print(bmatrix(Operator(circ).data))
