# https://github.com/cbarnes03/Quantum-Computing/blob/dd12b7276be7056dc0e36d3eeffaa561b04965cc/GroundState.py
#Learning how to generate the groundstate 

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])