# https://github.com/vallevaro/learning_quantum_computing/blob/a5736d91f3d1b0393616f14db460e102bea4802c/entanglement.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

# Create a quantum circuit with two qubits
qc = QuantumCircuit(2)

# Create an entangled state between the two qubits using the CNOT gate
qc.cx(0, 1)

# Add measurement for both qubits
qc.measure_all()

# Execute the circuit on a simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=8000)
result = job.result()

# Print the measurement results
counts = result.get_counts(qc)
print(counts)

# Plot a histogram of the measurement results
plot_histogram(counts)
