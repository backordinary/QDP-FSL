# https://github.com/andresserrato2004/deutsch-jozsa/blob/373dcae9b0fa2f3ee15408daa1fdca34261e1225/deutsch-josza.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
###### balanceada ###############

simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)
qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.x(3)
circuit.barrier()

circuit.barrier()
circuit.cx(2,3)
circuit.barrier()
circuit.h(2)
circuit.h(1)
circuit.h(0)
circuit.barrier()
circuit.measure(2, 2)
circuit.measure(0, 0)
circuit.measure(1, 1)

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()

### balanceada ################


simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)

qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.x(1)
circuit.barrier()
circuit.x(1)
circuit.barrier()
circuit.h(2)
circuit.h(1)
circuit.h(0)
circuit.barrier()
circuit.h(0)
circuit.barrier()

circuit.h(1)
circuit.h(0)
circuit.barrier()
circuit.measure(0, 1)
circuit.measure(1, 0)
compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()



## balanceada  ######### 
simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)


qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)



circuit.h(0)
circuit.h(1)
circuit.x(2)

circuit.barrier()
circuit.h(0)
circuit.x(1)
circuit.x(2)

circuit.barrier()
circuit.h(2)
circuit.h(1)
circuit.h(0)
circuit.barrier()
circuit.measure(0, 1)
circuit.measure(1, 0)

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()


 ### constante ###############

simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)
qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.x(3)
circuit.h(3)
circuit.barrier()




circuit.h(1)
circuit.h(0)
circuit.h(2)
circuit.barrier()

circuit.measure(0, 0)

circuit.measure(1, 1)

circuit.measure(2, 2)

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit)
plot_histogram(counts)
plt.show()
