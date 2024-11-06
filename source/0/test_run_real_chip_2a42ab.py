# https://github.com/busykoala/first_qiskit_steps/blob/5d4af47874317a7f40a3c1caa3bbb0976720bcae/test_run_real_chip.py
# Import Qiskit Terra
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ
from personal_credentials import QX_TOKEN, QX_URL

# Authenticate
IBMQ.enable_account(QX_TOKEN, QX_URL)

# Create a Quantum Register with 2 qubits.
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

# See a list of available devices.
print("IBMQ backends: ", IBMQ.backends())

# Compile and run the Quantum circuit on a device.
backend_ibmq = IBMQ.get_backend('ibmqx4')
job_ibmq = execute(qc, backend_ibmq)
result_ibmq = job_ibmq.result()

# Show the results.
print("real execution results: ", result_ibmq)
print(result_ibmq.get_counts(qc))
