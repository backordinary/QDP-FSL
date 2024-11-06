# https://github.com/Urinx/QuantumComputing/blob/3b2f9719ddca989bda4913db017f7728d47f9297/qiskit_codes/negative_superposition_state.py
# negative_superposition_state.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

# Define the Quantum and Classical Registers
q = QuantumRegister(1)
c = ClassicalRegister(1)

# Build the circuit
negative_superposition_state = QuantumCircuit(q, c)
negative_superposition_state.x(q)
negative_superposition_state.h(q)
negative_superposition_state.measure(q, c)

# Execute the circuit
job = execute(negative_superposition_state, backend = 'local_qasm_simulator', shots=1024)
result = job.result()

# Print the result
print(result.get_counts(negative_superposition_state))