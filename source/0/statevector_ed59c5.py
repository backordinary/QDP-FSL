# https://github.com/Aarun2/Quantum_Repo/blob/854234af2c4e14774ace90af5a4604507d4b1e50/Qiskit_Tutorials/Statevector.py
from qiskit import *

from qiskit.tools.visualization import plot_bloch_multivector

circuit = QuantumCircuit(1, 1)

circuit.x(0)

simulator = Aer.get_backend('statevector_simulator')

execute(circuit, backend=simulator)

result = execute(circuit, backend=simulator).result()
statevector = result.get_statevector()
print(statevector)

circuit = QuantumCircuit(1, 1)
circuit.x(0)
simulator = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend=simulator).result()
statevector = result.get_statevector()
print(statevector)

circuit.draw()

plot_bloch_multivector(statevector)

circuit.measure([0], [0])
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = backend, shots = 1024).result()
counts = result.get_counts()
from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)

circuit = QuantumCircuit(1, 1)
circuit.x(0)
simulator = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend=simulator).result()
statevector = result.get_unitary()
print(statevector)

circuit.draw()

circuit.measure([0], [0])
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = backend, shots = 1024).result()
counts = result.get_counts()
from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)

