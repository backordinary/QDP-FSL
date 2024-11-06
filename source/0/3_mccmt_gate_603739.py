# https://github.com/arshpreetsingh/Qiskit-cert/blob/ca4c9f6e1fe75f29c6ca3c9bc1d9d3e23dbf8813/3_MCCMT_gate.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.tools.monitor import job_monitor
from read_config import get_api_key

# Call MCMT from Qiskit
from qiskit.circuit.library import MCMT

# make connection
IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator')

# Now configure Circuit
q = QuantumRegister(6,'q')
c = ClassicalRegister(2,'c')
#  create a Quantum Circuit!
circuit = QuantumCircuit(q,c)

# Apply NOT Gates.
circuit.x(q[0])
circuit.x(q[1])
circuit.x(q[2])
circuit.x(q[3])
# now crate/apply MCMT
circuit += MCMT('h',4,2, label=None)
print("<<<<<<<<<<<<------------------------Circuit----------------->>>>>>>>>>>>.")
print(circuit)
circuit.measure(q[4],c[0])
circuit.measure(q[5],c[1])

job = execute(circuit, backend, shots=8192)
job_monitor(job)
counts = job.result().get_counts()
print(counts)