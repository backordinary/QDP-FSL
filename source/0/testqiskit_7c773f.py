# https://github.com/timothygao8710/Quantum-Computing-Course/blob/add45485678b80df2746491ad413ea32ac003275/QiskitExercises/testqiskit.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import Aer
from qiskit import IBMQ

qubits = QuantumRegister(3)
measurement = ClassicalRegister(3)
circuit = QuantumCircuit(qubits, measurement)
circuit.measure(qubits, measurement)

provider = IBMQ.load_account()

from qiskit.providers.ibmq import least_busy
backend = least_busy(provider.backends(filters=lambda x: int(x.configuration().n_qubits) >= 3 and 
                                   not x.configuration().simulator and x.status().operational==True))
print("least busy backend: ", backend)
# backend = Aer.get_backend('qasm_simulator')

qubits = QuantumRegister(3)
measurement = ClassicalRegister(3)
circuit = QuantumCircuit(qubits, measurement)

circuit.h(qubits)
circuit.ccx(qubits[0], qubits[1], qubits[2])
circuit.measure(qubits, measurement)

job = execute(circuit, backend, shots=10)
print(circuit)
result = job.result()
print(result)
# # Return counts
counts = result.get_counts(circuit)
# print("\nTotal count for 00 and 11 are:",counts)