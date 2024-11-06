# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/BuildGates/buildingGates.py
from qiskit import QuantumCircuit, Aer, execute
from math import pi

c_z = QuantumCircuit(2)
c_z.h(1)
c_z.cx(0, 1)
c_z.h(1)

c_y = QuantumCircuit(2)
c_y.sdg(1)
c_y.cx(0, 1)
c_y.s(1)

c_h = QuantumCircuit(2)
c_h.ry(pi/4, 1)
c_h.cx(0, 1)
c_h.ry(-pi/4, 1)

sw = QuantumCircuit(2)
sw.cx(0, 1)
sw.cx(1, 0)
sw.cx(0, 1)

print(c_z)
print(c_y)
print(c_h)
print(sw)