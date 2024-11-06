# https://github.com/austinjhunt/qiskit/blob/6f2135b16856d84544cd6ebd3e5064e322469afc/samples/sample.py
# import the qiskit library 
import qiskit
# prepare your circuit to run
from qiskit import IBMQ
 
# Qiskit quantum circuits libraries
# prepare a superposition of all possible computation states with 5 qubits
quantum_circuit = qiskit.circuit.library.QuantumVolume(5)
quantum_circuit.measure_all()
quantum_circuit.draw()
 
 
# Get the API token in
# https://quantum-computing.ibm.com/
IBMQ.save_account("YOUR TOKEN")
 
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_quito')
 
optimized_circuit = qiskit.transpile(quantum_circuit, backend)
optimized_circuit.draw()
# run in real hardware
job = backend.run(optimized_circuit)
retrieved_job = backend.retrieve_job(job.job_id())
result = retrieved_job.result()
print(result.get_counts())