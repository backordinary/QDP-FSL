# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/test_0_cloud.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer
from qiskit import IBMQ
from qiskit.backends.ibmq import least_busy
from qiskit.tools.visualization import circuit_drawer

# def available_functions():
#     print(IBMQ.active_accounts())
#     print(IBMQ.stored_accounts())

#Create a Quantum Register with 2 qubits.
q = QuantumRegister(2)
# Create a Classical Register with 2 bits.
c = ClassicalRegister(2)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)
# Add a H gate on qubit 0, putting this qubit in superposition.
qc.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(q[0], q[1])
# Add a Measure gate to see the state.
qc.measure(q, c)
# Draw the circuit
#circuit_drawer(qc)

# Print available backends
IBMQ.load_accounts()
print("Available backends:")
print("IBMQ backends: ", IBMQ.backends())

# Choose a real device with the least busy queue which can support our program (has at least 2 qubits).
# Alternatively, IBMQ.get_backend('backend_name')
large_enough_devices = IBMQ.backends(
    filters=
    lambda x: x.configuration()['n_qubits'] > 1 and not x.configuration()['simulator']
)
backend = least_busy(large_enough_devices)
print("The best backend is " + backend.name())

shots = 4096  # Number of shots to run the program (experiment); maximum is 8192 shots.
max_credits = 3  # Maximum number of credits to spend on executions.

job_exp = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
#If your experiment takes longer to run then you have time to wait around, or if you simply want to retrieve old jobs back, the IBMQ backends allow you to do that. First you would need to save your job's ID:
jobID = job_exp.job_id()
print("Your job id is " + jobID)

result_real = job_exp.result()
print("Results")
print(result_real.get_counts(qc))
