# https://github.com/arshpreetsingh/Qiskit-cert/blob/7946e8774dfa262264c5169bd8ef14ccb5e406e0/4_Diagonal_gate.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.circuit.library import Diagonal
from read_config import get_api_key
from qiskit.tools.monitor import job_monitor
# connecet with Circuit
# make connection
# make connection
IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator')


# Now configure the circuit!

q = QuantumRegister(2,'q')
c = QuantumRegister(2,'c')
# Create the circuit
circuit = QuantumCircuit(q,c)
diagonals = [-1,1,1,-1]
# create the circuit
circuit.h(q[0])
circuit.h(q[1])
circuit += Diagonal(diagonals)
circuit.h(q[0])
circuit.h(q[1])
print("<<<<<<<<<<<<------------------------Circuit----------------->>>>>>>>>>>>.")
print(circuit)
job = execute(circuit, backend, shots=8192)

job_monitor(job)
counts = job.result().get_counts()
print(counts)