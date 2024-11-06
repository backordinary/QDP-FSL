# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/1_build_circuit.py
# Do Required imports!
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.tools.monitor import job_monitor
from read_config import get_api_key
# Make Connection with IBM Computer!
IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator')
print(backend)
from qiskit.quantum_info import Statevector

#Start Quantum-programming!
q = QuantumRegister(1, 'q')
c = ClassicalRegister(1, 'c')
# Create Circuit and add required Gates!
circuit = QuantumCircuit(q, c)
circuit.x(q[0])               #-------Applying Pauli-X Here on Qubit[0].
circuit.measure(q, c)
print(circuit)
# Execute things on Quantum-Computer!
job = execute(circuit, backend=backend, shots=1)
job_monitor(job)
#print("Job_Results-->>",job.result())
counts = job.result().get_counts()
print("counts", counts)
print("Completed!")
