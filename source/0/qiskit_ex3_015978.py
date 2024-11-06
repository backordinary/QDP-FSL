# https://github.com/mohsenhariri/qisirq/blob/9b6e839365a87c099f85d8b97540951b5dedd63d/pkg/qiskit_ex3.py
print(__file__)
import numpy as np
from qiskit import QuantumCircuit, transpile

# QasmSimulator: is the Aer high performance circuit simulator.
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# QuantumCircuit: can be thought as the instructions of the quantum system. It holds all your quantum operations.


simulator = QasmSimulator()

# I am initializing with 2 qubits in the 0 state.
# With 2 classical bits set to 0
circuit = QuantumCircuit(3, 2)

### ADDING GATES (OPERATIONS) ###
circuit.h(qubit=0)
circuit.h(qubit=2)
circuit.h(qubit=2)
circuit.h(qubit=2)

circuit.cx(control_qubit=0, target_qubit=1)

circuit.measure([0, 1], [0, 1])


compiled_circuit = transpile(circuits=circuit, backend=simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(compiled_circuit)

print(counts)

print(circuit.draw())
print("ger")
exit()
import matplotlib.pyplot as plt

plot_histogram(counts)
plt.show()
