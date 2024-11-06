# https://github.com/ole-knf/Einsatz-von-Quantencomputern-zur-Abschwaechung-klassischer-Verfahren-der-Kryptographie/blob/70c5297235927b5f784c63b19d1009b89ed86d78/Introduction%20Qiskit.py
# The following code is based on https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html
from qiskit import *


# Create a quantum circuit acting on a quantum register of three qubits
circ = QuantumCircuit(3)
# Add one Hadamard gate and two CNOT gates
circ.h(0)
circ.cx(0, 1)
circ.cx(0, 2)
# Create a visual output of the circut
print(circ)

# Create another quantum circuit with three qubits and three classical bits
meas = QuantumCircuit(3, 3)
# Use a barrier to keep future optimizations within the circut
meas.barrier(range(3))
# Add a mesurement for every qubit
meas.measure(range(3), range(3))
# Combine the new circut with the old one
circ.add_register(meas.cregs[0])
qc = circ.compose(meas)
print(qc)

# Use Aer's qasm_simulator to simulate the circuit with 10000 iterations
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = backend_sim.run(transpile(qc, backend_sim), shots=10000)
result_sim = job_sim.result()
# Sum up all mesurements and return the distribution
counts = result_sim.get_counts(qc)
print(counts)
