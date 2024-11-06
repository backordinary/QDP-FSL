# https://github.com/vallevaro/learning_quantum_computing/blob/b16c1fd0ba525c4e546dd8658d2f20e2566183af/hello_world.py
from qiskit import QuantumCircuit, execute, Aer

# Create a quantum circuit with one qubit and one classical bit
qc = QuantumCircuit(1, 1)

# Add a "Hello, World!" message to the classical bit
qc.measure(0, 0)

# Execute the circuit on a simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend)
result = job.result()

# Print the measurement result
print(result.get_counts(qc))
