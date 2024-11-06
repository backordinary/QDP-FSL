# https://github.com/39xdgy/selflearn_qiskit/blob/874d6aceb8487ec1cd66f5da7c516d54ba541bab/gates.py
from qiskit import *
from qiskit.tools.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
circuit = QuantumCircuit(1, 1)
circuit.x(0)
simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend = simulator).result()
statevector = result.get_statevector()
print(statevector)
circuit.draw(output = 'mpl')



plot_bloch_multivector(statevector)

circuit.measure([0], [0])
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = backend, shots = 1024).result()

counts = result.get_counts()

from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)

plt.show()


circuit = QuantumCircuit(1, 1)
circuit.x(0)
simulator = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend = simulator).result()
unitary = result.get_unitary()
print(unitary)
