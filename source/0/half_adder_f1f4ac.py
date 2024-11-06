# https://github.com/Timgrau/Projektarbeit-2/blob/52424ec38125f8d3d9106f633908b12ea033854d/ausarbeitung/code/half-adder.py
from qiskit import QuantumRegister,\
    ClassicalRegister, QuantumCircuit

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[1])

circuit.measure(qreg_q[2], creg_c[1])
circuit.measure(qreg_q[1], creg_c[0])
