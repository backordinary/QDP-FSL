# https://github.com/AdamNaoui/LOG6953C_TPs/blob/f717a67610b4dad05190677d698d3553af76d46e/TP2/grover_ibmq.py
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister, execute, IBMQ

from qiskit_aer import QasmSimulator
from qiskit.visualization import plot_histogram
import qiskit.quantum_info as qi

from oracle import oracle

IBMQ.save_account(
    'e31a3c8ff39a7a22b6a3052adfb2d4d150b53f5bcac9b161fefba1349302c57ffbf55da2b310ecd734c74be5874804be20d4c1cc4e0345e74db6569e9989fab8',
    overwrite=True)
provider = IBMQ.load_account()
backend = provider.backends.simulator_extended_stabilizer

target = '01101'
curr_oracle = oracle(target)

# creating inversion about mean operator
A = [[1 / (2 ** len(target)) for i in range(2 ** len(target))] for j in range(2 ** len(target))]
# Convert to NumPy matrix
np_a = np.array(A)
I = np.identity(2 ** len(target))
matrix = -I + 2 * np_a

# Use Aer's qasm_simulator
simulator = QasmSimulator()
# Create quantum program that find 01101 by reversing its phase
x = QuantumRegister(len(target), name='x')  # 5 qubits index 0 is the right most qubit
fx = QuantumRegister(1, name='ands_results')
res = ClassicalRegister(len(target), name='Target')

grover = QuantumCircuit(x, res, fx, name='grover')
grover.h(x)

for j in range(int((2 ** len(target)) ** (1 / 2))):
    grover.append(curr_oracle, x[:] + fx[:])
    grover.x(fx)
    inversion_about_mean = qi.Operator(matrix.tolist())
    grover.unitary(inversion_about_mean, x, label='Inversion about mean')

# Map the quantum measurement to the classical bits
grover.measure(x, res)

# compiled_circuit = transpile(grover, simulator)
transpiled = transpile(grover, backend=backend)
job = backend.run(transpiled, shots=20000)
# Execute the circuit on the qasm simulator

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(grover)

# Draw the circuit
grover.draw('mpl', filename='grover_ibmq.png')

plot_histogram(counts, filename='grover_ibmq_hist.png', title='Grover IBMQ Histogram', bar_labels=True, figsize=(10, 8))
