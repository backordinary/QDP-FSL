# https://github.com/LauritaGutierrez/proyecto_final/blob/94e48164dc3585491ae647dbba08eb53f6d6b214/Deustch%20Jozsa/Algoritmo_deustch_Jozsa.py
#Librerias necesarias para ejecución del codigo

from qiskit import QuantumCircuit, transpile
from qiskit import Aer
import Functions as Func
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')

# 1 implementación de algoritmo

circ = QuantumCircuit(4, 3)
circ.x(3)
circ.barrier(range(4))
circ.h(range(4))
circ.barrier(range(4))
circ.i(range(4))
circ.barrier(range(4))
circ.h((0, 1))
circ.barrier(range(4))
circ.measure((0, 1), (0, 1))

compil_circ= transpile(circ, simulator)
comienzo = simulator.run(compil_circ, desde = 1000)
result = comienzo.result()
counts = result.get_counts(circ)
print(circ)
plt.show()
plot_histogram(counts)


# 2 implementación del algoritmo

circ = QuantumCircuit(4, 3)
circ.x(3)
circ.barrier(range(4))
circ.h(range(4))
circ.barrier(range(4))
circ.cx(0, 3)
circ.barrier(range(4))
circ.h((0, 1))
circ.barrier(range(4))
circ.measure((0, 1), (0, 1))
compil_circ = transpile(circ, simulator)
comienzo = simulator.run(compil_circ, desde = 1000)
result = comienzo.result()
counts = result.get_counts(circ)
print(circ) # el resultado será 0000 o 1111
plt.show()
plot_histogram(counts)

