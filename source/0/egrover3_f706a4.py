# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/egrover/egrover3.py
# initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit

# import basic plot tools
from qiskit.visualization import plot_histogram

# Run our circuit on the least busy backend. Monitor the execution of the job in the queue
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor


# --------------------- Initializing the 3-qubit circuit ----------------------------------

qc = QuantumCircuit(3)
qc.cz(0, 2)
qc.cz(1, 2)
qc.draw(output="mpl")

oracle_ex3 = qc.to_gate()
oracle_ex3.name = "U$_\omega$"


# --------------------- Generalized n-qubit diffuser --------------------------------------


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


# --------------------- Building the complete grover circuit -------------------------------


def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc


n = 3
grover_circuit = QuantumCircuit(n)
grover_circuit = initialize_s(grover_circuit, [0, 1, 2])

num_iterations = 1

for i in range(0, num_iterations):
    grover_circuit.append(oracle_ex3, [0, 1, 2])
    grover_circuit.append(diffuser(n), [0, 1, 2])

grover_circuit.measure_all()
grover_circuit.draw(output="mpl")
plt.show()


# --------------------- Simulation ---------------------------------------------------------

aer_sim = Aer.get_backend("aer_simulator")
transpiled_grover_circuit = transpile(grover_circuit, aer_sim)
qobj = assemble(transpiled_grover_circuit)
results = aer_sim.run(qobj).result()
counts = results.get_counts()
plot_histogram(counts)
# plt.savefig("./results.png")
plt.show()


# --------------------- Computation of quantum hardware ------------------------------------
# provider = IBMQ.load_account()
# provider = IBMQ.get_provider("ibm-q")
# backend = least_busy(
#     provider.backends(
#         filters=lambda x: x.configuration().n_qubits >= 3
#         and not x.configuration().simulator
#         and x.status().operational == True
#     )
# )
# print("Least busy backend: ", backend)


# transpiled_grover_circuit = transpile(grover_circuit, backend, optimization_level=3)
# job = backend.run(transpiled_grover_circuit)
# job_monitor(job, interval=2)

# # Get the results from the computation
# results = job.result()
# print("Results: ", results)
# answer = results.get_counts(grover_circuit)
# print("Answer: ", answer)
# plot_histogram(answer)
# plt.savefig("./results.png")
# plt.show()
