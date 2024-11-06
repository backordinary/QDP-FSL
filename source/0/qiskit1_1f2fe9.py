# https://github.com/cccbook/aipy/blob/6bb642b5f308033e933996052600af9e37dccd7a/08-%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86/19e-chatGPT/34-quantum/qiskit1.py
# Import the required libraries
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

# Define a quantum circuit with two qubits and two classical bits
q = QuantumRegister(2)
c = ClassicalRegister(2)
circuit = QuantumCircuit(q, c)

# Apply a Hadamard gate to the first qubit
circuit.h(q[0])

# Apply a CNOT gate with the first qubit as the control and the second qubit as the target
circuit.cx(q[0], q[1])

# Measure both qubits
circuit.measure(q, c)

# Execute the circuit on a quantum simulator
backend = 'local_qasm_simulator'
job = execute(circuit, backend)
result = job.result()

# Print the result
print(result.get_counts(circuit))
