# https://github.com/BHazel/hello-quantum/blob/a0578d086df28c0ee66c686a4bddc0ec285eb464/concepts/superposition/qiskit/main.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute

print('*** Hello, Quantum! - Superposition (Qiskit) ***')

# Set quantum and classical bit registers.
repeat_count = 1000
q = QuantumRegister(1, 'q')
c = ClassicalRegister(1, 'c')

# Set up cirbuit with quantum and classical bit registers.
# Apply Hadamard gate to create a superposition.
# Measure the qubit and save the result into a classical bit.
circuit = QuantumCircuit(q, c)
circuit.h(q[0])
circuit.measure(q, c)

# Initialise simulator and run circuit a specified number of times.
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=repeat_count)

# Get counts of qubit measurement results.
result_counts = job.result().get_counts()

# Print results.
print(f"Counts for {repeat_count} repeats:")
print(f"\t0: {result_counts['0']}")
print(f"\t1: {result_counts['1']}")