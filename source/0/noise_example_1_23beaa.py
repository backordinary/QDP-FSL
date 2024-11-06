# https://github.com/vallevaro/learning_quantum_computing/blob/a5736d91f3d1b0393616f14db460e102bea4802c/noise_example_1.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer import noise

# Create a quantum circuit with one qubit
qc = QuantumCircuit(1)

# Apply a random single qubit gate
qc.rx(0.5)

# Add a measurement
qc.measure_all()

# Create a noise model for the simulator
error_model = noise.NoiseModel()

# Add random single qubit errors to the noise model
error_model.add_all_qubit_quantum_error(noise.random_unitary(0.05, 1))

# Execute the circuit on a simulator with noise
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, noise_model=error_model, shots=8000)
result = job.result()

# Print the measurement results
counts = result.get_counts(qc)
print(counts)
