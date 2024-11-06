# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/picture_circuits/ciruits.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.tools.visualization import circuit_drawer  # plot_state_qsphere

n = 2
q = QuantumRegister(n, 'q')
c = ClassicalRegister(n, 'c')
qc = QuantumCircuit(q, c)

qc.h(q)
qc.x(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)

circuit_drawer(qc, filename="./sample_circuit.png")
