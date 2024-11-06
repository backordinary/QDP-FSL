# https://github.com/takehuge/Qalgorithm/blob/3f26c624ff38e03e915b3949d583cb747d73aca3/QisKit/RunQIS01.py
# use IBMQ.save_account('MY_API_TOKEN') on Terminal to save token on local drive first

import numpy as np
from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute, IBMQ
from qiskit.backends.ibmq import least_busy
from qiskit.tools.visualization import circuit_drawer, plot_state, plot_histogram                  

# 1: Qiskit Terra (Basic Quantum Algorithm Assembly)

# 1.1: Building the Circuit
q = QuantumRegister(3) # Create a Quantum Register with 3 qubits.
circ = QuantumCircuit(q) # Create a Quantum Circuit acting on the q register
circ.h(q[0]) # Add a H gate on qubit 0, putting this qubit in superposition.
# Put the qubits 0, 1 in a Bell state.
circ.cx(q[0], q[1]) # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
# Put the qubits 0, 1, 2 in a GHZ state.
circ.cx(q[0], q[2]) # Add a CX (CNOT) gate on control qubit 0 and target qubit 2

# 1.2: Visualize the Circuit
circuit_drawer(circ, filename="Circuit.png").show()

# 2: Qiskit Aer (For Simulation purposes like QuTiP)

# 2.1: Run the quantum circuit on a statevector simulator backend
backend = Aer.get_backend('statevector_simulator')
job = execute(circ, backend) # Create a Quantum Program for execution 
result = job.result()
outputstate = result.get_statevector(circ)
print("simulation: ", result )
print(np.around(outputstate,3))
plot_state(outputstate)

# 2.2: Run the quantum circuit on a unitary simulator backend
backend = Aer.get_backend('unitary_simulator')
job = execute(circ, backend)
result = job.result()
print("simulation: ", result )
print(np.around(result.get_unitary(circ), 3))

# 2.3: OpenQASM backend (QASM: Qiskit-Aer Simulating Measurement)
# To simulate a circuit that includes measurement, we need to add measurements 
# to the original circuit above, and use a different Aer backend.
c = ClassicalRegister(3, 'c') # Create a Classical Register with 3 bits.
meas = QuantumCircuit(q, c) # Create a Quantum Circuit
meas.barrier(q) # Create barrier figuratively? between operations & measurements
meas.measure(q,c) # map the quantum measurement to the classical bits
# The Qiskit circuit object supports composition using the addition operator.
qc = circ + meas
#drawing the circuit
# circuit_drawer(qc)
backend_sim = Aer.get_backend('qasm_simulator') # Use Aer's qasm_simulator
job_sim = execute(qc, backend_sim, shots=1024) # shots: # of repeats of the circuit (default: 1024)
result_sim = job_sim.result() # Grab the results from the job.
counts = result_sim.get_counts(qc)
print(counts)
plot_histogram(counts)

# 3: IBM Q

# 3.1: Running circuits on real devices
# load IBM Q user-specific API-TOKEN
IBMQ.load_accounts()
s = IBMQ.backends()
print("Available backends:")
print(s)
# Select the least busy real Qubit
criteria = IBMQ.backends(filters=lambda x: x.configuration()['n_qubits'] > 3 
        and not x.configuration()['simulator'])
backend = IBMQ.get_backend("ibmqx4") #least_busy(criteria)
# print("The best backend is " + backend.name()) 
print("Running on %s" %backend.name())
shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
max_credits = 3        # Maximum number of credits to spend on executions. 
job_exp = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
result_exp = job_exp.result()
counts_exp = result_exp.get_counts(qc)
plot_histogram(counts_exp)

# 3.2: Simulating circuits using a HPC simulator
backend = IBMQ.get_backend('ibmq_qasm_simulator')
shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
max_credits = 3        # Maximum number of credits to spend on executions. 
job_hpc = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
result_hpc = job_hpc.result()
counts_hpc = result_hpc.get_counts(qc)
plot_histogram(counts_hpc)

# Retrieving a previously ran job
jobID = job_exp.job_id()
print('JOB ID: {}'.format(jobID))
job_get=backend.retrieve_job(jobID)
job_get.result().get_counts(qc)


