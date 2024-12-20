# https://github.com/Nat15005/DeutschYDeutsch-Jozsa/blob/07c15bcc325bfc04979230769f4075e1124fb0b9/Funciones_DEUTSCH/Constante_0/Constante_0_(1,1).py
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(2, 2)

circuit.x(0)
circuit.x(1)
circuit.id(0)
circuit.id(1)

# Map the quantum measurement to the classical bits
circuit.measure([0, 1], [1, 0])

compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:", counts)
print(circuit)
plot_histogram(counts)
plt.show()