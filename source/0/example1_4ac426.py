# https://github.com/juancsosap/pythontraining/blob/1cc3ddae18055969477678e1b4e689e215c175fa/projects/quantum/example1.py
# sudo pip3 install qiskit

# Quantum Information System Kit
from qiskit import IBMQ
#IBMQ.save_account('5d31d26be51bead8a08877bb279fb1019f95e824e385b5acb656547c87a1abb3902593932264148fd73ddae2c1cc2fdfe06e1a66b1c2b5e206e0d873591b27f0')

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute

# Import Aer
from qiskit import BasicAer

from qiskit.tools.visualization import plot_state_city, plot_histogram
from qiskit.tools.monitor import job_monitor

# Create a Quantum Register with 3 qubits.
q = QuantumRegister(3, 'q')

# Create a Quantum Circuit acting on the q register
circ = QuantumCircuit(q)

# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(q[0], q[1])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(q[0], q[2])

circ.draw(output='mpl')

# Run the quantum circuit on a statevector simulator backend
backend = BasicAer.get_backend('statevector_simulator')

# Create a Quantum Program for execution
job = execute(circ, backend)

result = job.result()

outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

plot_state_city(outputstate)

# Run the quantum circuit on a unitary simulator backend
backend = BasicAer.get_backend('unitary_simulator')
job = execute(circ, backend)
result = job.result()

# Show the results
print(result.get_unitary(circ, decimals=3))

# Create a Classical Register with 3 bits.
c = ClassicalRegister(3, 'c')
# Create a Quantum Circuit
meas = QuantumCircuit(q, c)
meas.barrier(q)
# map the quantum measurement to the classical bits
meas.measure(q,c)

# The Qiskit circuit object supports composition using
# the addition operator.
qc = circ+meas

#drawing the circuit
qc.draw(output='mpl')

# Use Aer's qasm_simulator
backend_sim = BasicAer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = execute(qc, backend_sim, shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()

counts = result_sim.get_counts(qc)
print(counts)

plot_histogram(counts)

IBMQ.load_accounts()

print("Available backends:")
IBMQ.backends()

from qiskit.providers.ibmq import least_busy

large_enough_devices = IBMQ.backends(filters=lambda x: x.configuration().n_qubits > 3 and not x.configuration().simulator)
backend = least_busy(large_enough_devices)
print("The best backend is " + backend.name())

shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
max_credits = 3        # Maximum number of credits to spend on executions.

job_exp = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
job_monitor(job_exp)

result_exp = job_exp.result()

counts_exp = result_exp.get_counts(qc)
plot_histogram([counts_exp,counts])

backend = IBMQ.get_backend('ibmq_qasm_simulator', hub=None)

shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
max_credits = 3        # Maximum number of credits to spend on executions.

job_hpc = execute(qc, backend=backend, shots=shots, max_credits=max_credits)

result_hpc = job_hpc.result()

counts_hpc = result_hpc.get_counts(qc)
plot_histogram(counts_hpc)

jobID = job_exp.job_id()

print('JOB ID: {}'.format(jobID))

job_get=backend.retrieve_job(jobID)

job_get.result().get_counts(qc)
