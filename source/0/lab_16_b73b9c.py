# https://github.com/sujitmandal/Quantum-Programming/blob/1d348c307584bb804255eab75357789786ac5437/The%20Coding%20School%20Lab/Lab_16.py
import json
import matplotlib.pyplot as plt
from qiskit import __qiskit_version__, QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_bloch_multivector

print('Qiskit Version :')
print(json.dumps(__qiskit_version__, indent=3))


circ = QuantumCircuit(1)  
circ.draw('mpl') 
plt.show()


circ_bell = QuantumCircuit(2)
circ_bell.h(0)
circ_bell.cx(0, 1)
circ_bell.draw('mpl')
plt.show()


backend = Aer.get_backend('statevector_simulator') 
job = execute(circ_bell, backend)
result = job.result()
state_bell = result.get_statevector(circ_bell)
print(state_bell) 
counts = result.get_counts(circ_bell)
print(counts)

plot_histogram(counts)
plt.show()