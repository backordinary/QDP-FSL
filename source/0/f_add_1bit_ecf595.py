# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/f_add_1bit.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.visualization import plot_histogram
import sys


# Add q[1] and q[2], put the result in q[1]q[2]
# N.B.: in the output the most significant bit (q[2] in this case) is on the left
def addition_1bit(circuit, q):
    circuit.h(q[2])
    circuit.cx(q[1], q[2])
    circuit.tdg(q[2])
    circuit.cx(q[0], q[2])
    circuit.t(q[2])
    circuit.cx(q[1], q[2])
    circuit.tdg(q[2])
    circuit.cx(q[0], q[2])
    circuit.t(q[2])
    circuit.h(q[2])
    circuit.t(q[1])
    circuit.cx(q[0], q[1])
    circuit.t(q[0])
    circuit.tdg(q[1])


# n-qubit number input state
def number_state(circuit, q, a, b):
    if a == 1:
        circuit.x(q[0])  # q[0] contains the value of a
    if b == 1:
        circuit.x(q[1])  # q[1] contain the value of b


qr = QuantumRegister(3)
cr = ClassicalRegister(3)
qc = QuantumCircuit(qr, cr)

a = 0
b = 0

if (sys.argv[1] == '1'):
    a = 1
if (sys.argv[2] == '1'):
    b = 1

number_state(qc, qr, a, b)
addition_1bit(qc, qr)

qc.measure(qr, cr)
circuit_drawer(qc, filename="imgs/f_add_1bit.png")

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend=backend)
result = job.result()
counts = result.get_counts(qc)
print(counts)
plot_histogram(counts)
