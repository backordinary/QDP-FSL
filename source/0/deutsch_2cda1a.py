# https://github.com/andresserrato2004/deutsch-jozsa/blob/373dcae9b0fa2f3ee15408daa1fdca34261e1225/deutsch.py

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)
circuit.x(0)
circuit.x(0)

circuit.barrier()
circuit.x(0)
circuit.cx(0, 1)
circuit.x(0)
circuit.barrier()


circuit.measure(0,1)
circuit.measure(1,0)

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()



##########################
simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)

circuit.x(0)
circuit.x(0)
circuit.barrier()
circuit.x(0)
circuit.barrier()
circuit.measure(0,1)
circuit.measure(1,0)


compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()


###############
simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)

circuit.x(1)
circuit.barrier()
circuit.h(0)
circuit.h(1)

circuit.barrier()

circuit.x(1)
circuit.barrier()

circuit.h(0)
circuit.barrier()
circuit.measure(0,1)
circuit.measure(1,0)

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()

########

simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)

circuit.barrier()
qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)
circuit.x(0)
circuit.x(1)
circuit.barrier()
circuit.id(0)
circuit.x(1)
circuit.barrier()

circuit.measure(0,1)
circuit.measure(1,0)


compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()