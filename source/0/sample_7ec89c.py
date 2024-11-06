# https://github.com/SamYuan1990/Trash/blob/28e0142c122fc35b21f6ef3372f703c8329743e0/qiskit/sample.py
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Use Aer's qasm_simulator
simulator = AerSimulator()

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
# compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
# job = simulator.run(compiled_circuit, shots=1000)

# Grab results from the job
# result = job.result()

result_ideal = qiskit.execute(circuit, simulator).result()
counts_ideal = result_ideal.get_counts(0)

# Returns counts
print("\nTotal count for 00 and 11 in 1000 times are:",counts_ideal)