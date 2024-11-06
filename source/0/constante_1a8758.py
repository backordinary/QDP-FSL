# https://github.com/Nat15005/DeutschYDeutsch-Jozsa/blob/07c15bcc325bfc04979230769f4075e1124fb0b9/Deutsch-Jozsa/Constante.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import matplotlib.pyplot as plt

"========================= Funciones Deutsch - Jozsa (Constante) ========================="

simulator = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(5, 5)

circuit.x(4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)
circuit.barrier()

circuit.id(0)
circuit.id(1)
circuit.id(2)
circuit.id(3)
circuit.id(4)

circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.barrier()

circuit.measure([0,1,2,3,4], [4,3,2,1,0])
compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)

print("\nTotal count for 00 and 11 are:", counts)
print(circuit)
plot_histogram(counts)
plt.show()




