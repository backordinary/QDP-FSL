# https://github.com/LauritaGutierrez/proyecto_final/blob/94e48164dc3585491ae647dbba08eb53f6d6b214/Deutsch/algoritmo_deutsch.py
#Librerias necesarias para ejecución del codigo

from qiskit import QuantumCircuit, transpile
from qiskit import Aer
import Functions as Func
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')

# Implementación del primer algoritmo

circ = QuantumCircuit(3, 2)
circ.x(2)
circ.barrier(0,2)
circ.h(0)
circ.h(2)
circ.barrier(0,2)
circ.id(0)
circ.id(2)
circ.barrier(0,2)
circ.h(0)
circ.barrier(0,2)
circ.measure([0], [0])

compil_circ = transpile(circ, simulator)
comienzo = simulator.run(compil_circ, shots=1000)
result = comienzo.result()
counts = result.get_counts(circ)
print(circ)
plt.show()
plot_histogram(counts)


# Implementación del segundo algoritmo

circ = QuantumCircuit(3, 2)
circ.x(2)
circ.barrier()
circ.h(0)
circ.h(2)
circ.barrier(0,2)
circ.x(0)
circ.cnot(0, 2)
circ.x(0)
circ.barrier(0,2)
circ.h(0)
circ.barrier(0,2)
circ.measure([0], [0])

compil_circ = transpile(circ, simulator)
comienzo = simulator.run(compil_circ, desde = 1000)
fin = comienzo.result()
counts =fin.get_counts(circ)
print(circ)
plt.show()
plot_histogram(counts)

# Implementación del tercer algoritmo

circ = QuantumCircuit(3, 2)
circ.x(2)
circ.barrier(0,2)
circ.h(0)
circ.h(2)
circ.barrier(0,2)
circ.x(2)
circ.barrier(0,2)
circ.h(0)
circ.barrier(0,2)
circ.measure([0], [0])

compil_circ = transpile(circ, simulator)
comienzo = simulator.run(compil_circ, desde =1000)
result = comienzo.result()
counts = result.get_counts(circ)
print(circ)
plt.show()
plot_histogram(counts)

# Implementación del cuarto algoritmo
circuit = QuantumCircuit(3, 2)
circ.x(2)
circ.barrier(0,2)
circ.h(0)
circ.h(2)
circ.barrier(0,2)
circ.x(0)
circ.cnot(0, 2)
circ.x(0)
circ.barrier(0,2)
circ.h(0)
circ.barrier(0,2)
circ.measure([0], [0])

compiled_circuit = transpile(circ, simulator)
comienzo = simulator.run(compiled_circuit, desde = 1000)
result = comienzo.result()
counts = result.get_counts(circ)
print(circ)
plt.show()
plot_histogram(counts)