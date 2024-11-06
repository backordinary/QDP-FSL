# https://github.com/Phystro/quantum_computing/blob/c8ab9d32d46a9fbab0898c274b2ff3e1a370744c/qiskit/multiple_qubit_gates/main.py
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# build a 3 qubit quantum register
qr = QuantumRegister(3)
# build a 3 bit classical register
cr = ClassicalRegister(3)
# quantum circuit
qc = QuantumCircuit(qr, cr)
qc.h(0)
qc.h(1)
qc.h(2)

#qc.draw()
print(qc)

# Simulate output
# simulator = Aer.get_backend('qasm_simulator')
# result = execute(qc, backend=simulator, shots=1024).result()
# counts = result.get_counts()
# print(counts)
