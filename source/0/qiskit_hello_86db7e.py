# https://github.com/Quantum179/Physics-Playground/blob/0926e62eb638975cb8325f727629500f0bb30f7a/Quantum%20Computing/qiskit_hello.py
# source : https://qiskit.org/documentation/getting_started.html

import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import BasicAer, Aer

from qiskit.tools.visualization import plot_state_city
from qiskit.tools.visualization import plot_histogram

from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

q = QuantumRegister(3, 'q')

circ = QuantumCircuit(q)

circ.h(q[0])
circ.cx(q[0], q[1])
circ.cx(q[0], q[2])

# %matplotlib inline
circ.draw(output='mpl')

# Run the quantum circuit on a unitary simulator backend
backend = BasicAer.get_backend('statevector_simulator')

job = execute(circ, backend)
result = job.result()

outputstate = result.get_statevector(circ, decimals=3)

print(outputstate)
plot_state_city(outputstate)

# Run the quantum circuit on a unitary simulator backend
backend = Aer.get_backend('unitary_simulator')

job = execute(circ, backend)
result = job.result()

print(result.get_unitary(circ, decimals=3))


# Run the quantum circuit with measurements
c = ClassicalRegister(3, 'c')

meas = QuantumCircuit(q, c)
meas.barrier(q)
meas.measure(q, c)

qc = circ + meas

# %matplotlib inline
qc.draw(output='mpl')


# Run with qasm simulator
backend_sim = BasicAer.get_backend('qasm_simulator')

job_sim = execute(qc, backend_sim, 1024)
result_sim = job_sim.result()

counts = result_sim.get_counts(qc)

print(counts)
plot_histogram(counts)

# Run on a IBMQ calculator
IBMQ.load_accounts()

print("Available backends:")
IBMQ.backends()

large_enough_devices = IBMQ.backends(filters=lambda x: x.configuration().n_qubits > 3 and not x.configuration().simulator)
backend = least_busy(large_enough_devices)

print("The best backend is " + backend.name())

job_exp = execute(qc, backend, shots=1024, max_credits=3)
job_monitor(job_exp)

result_exp = job_exp.result()

counts_exp = result_exp.get_counts(qc)
plot_histogram([counts_exp,counts])


