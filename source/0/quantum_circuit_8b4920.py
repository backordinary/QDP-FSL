# https://github.com/sujitmandal/Quantum-Programming/blob/1d348c307584bb804255eab75357789786ac5437/quantum_circuit.py
import qiskit
from qiskit import Aer
from qiskit import execute
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import plot_bloch_multivector

'''
This programe is create by Sujit Mandal
Github: https://github.com/sujitmandal
Pypi : https://pypi.org/user/sujitmandal/
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
'''

circuit = QuantumCircuit(2, 2)
circuit.x(0)

simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend=simulator).result()
statevector = result.get_statevector()
print(statevector)

circuit.draw(output='mpl')
plt.show()

plot_bloch_multivector(statevector)
plt.show()

circuit.measure([0], [0])
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=backend, shots=1024).result()
counts = result.get_counts()

plot_histogram(counts)
plt.show()