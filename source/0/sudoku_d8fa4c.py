# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/sudoku.py
# initializations
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools
from qiskit.visualization import plot_histogram

# --------------------- Utilities ---------------------------------------------------------
def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits - 1)
    qc.mct(list(range(nqubits - 1)), nqubits - 1)  # multi-controlled-toffoli
    qc.h(nqubits - 1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s


def XOR(qc, a, b, output):
    qc.cx(a, output)
    qc.cx(b, output)


# --------------------- Building oracle to check clauses ---------------------------------------------------------
clause_list = [[0, 1], [0, 2], [1, 3], [2, 3]]

var_qubits = QuantumRegister(4, name="v")
clause_qubits = QuantumRegister(4, name="c")
output_qubit = QuantumRegister(1, name="out")
cbits = ClassicalRegister(4, name="cbits")
qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit, cbits)


def sudoku_oracle(qc, clause_list, clause_qubits):
    # compute clauses with XOR
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1

    # flip 'output' bit if all clauses are satisfied
    qc.mct(clause_qubits, output_qubit)

    # uncompute clauses to reset clause-checking bits to 0
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1


# --------------------- Building the complete circuit ---------------------------------------------------------

# initialize 'out0' in state |->
qc.initialize([1, -1] / np.sqrt(2), output_qubit)

# initialize qubits in state |s>
qc.h(var_qubits)
qc.barrier()  # for visual separation

# full Grover procedure
num_iterations = 2
for i in range(0, num_iterations):
    # apply oracle
    sudoku_oracle(qc, clause_list, clause_qubits)
    qc.barrier()  # for visual separation
    # apply diffuser
    qc.append(diffuser(4), [0, 1, 2, 3])

# measure the variable qubits
qc.measure(var_qubits, cbits)

qc.draw(fold=-1, output="mpl")
plt.show()

# --------------------- Simulation ---------------------------------------------------------
aer_sim = Aer.get_backend("aer_simulator")
transpiled_qc = transpile(qc, aer_sim)
result = aer_sim.run(transpiled_qc).result()
plot_histogram(result.get_counts())
plt.show()
