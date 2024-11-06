# https://github.com/mohsenhariri/qisirq/blob/9b6e839365a87c099f85d8b97540951b5dedd63d/pkg/qis_tut_circuit_basic.py
# https://qiskit.org/documentation/tutorials/circuits/01_circuit_basics.html
# https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html

import matplotlib.pylab as plt

# from qiskit.circuit import QuantumCircuit
from qiskit import QuantumCircuit

# QuantumCircuit(4, 3)  A QuantumCircuit with 4 qubits and 3 classical bits
qc = QuantumCircuit(4, 3)
print(qc.draw())

qc.h(0)
print(qc.draw())

qc.cx(control_qubit=1, target_qubit=2)
print(qc.draw())

qc.cx(control_qubit=0, target_qubit=2)
print(qc.draw())

qc.measure_all()
print(qc.draw())

# qc.draw('mpl')
# plt.show()


from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

qr1 = QuantumRegister(2, "q1")
qr2 = QuantumRegister(5, "q2")
cr = ClassicalRegister(5, "c")


qc = QuantumCircuit(qr1, qr2, cr)
qc.h(qr1[:])
qc.h(qr2[1:3])
print(qc.draw())

qc.measure(qr2, cr)
print(qc.draw())
