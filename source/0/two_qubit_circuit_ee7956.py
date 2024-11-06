# https://github.com/sujitmandal/Quantum-Programming/blob/1d348c307584bb804255eab75357789786ac5437/two_qubit_circuit.py
import qiskit
from qiskit import Aer
from qiskit import execute
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram

'''
This programe is create by Sujit Mandal
Github: https://github.com/sujitmandal
Pypi : https://pypi.org/user/sujitmandal/
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
'''

qr = QuantumRegister(2)
cr = ClassicalRegister(2)

circuit = QuantumCircuit(qr, cr)
print(circuit.draw())

circuit.h(qr[0])
circuit.draw(output='mpl')
plt.show()

circuit.cx(qr[0], qr[1]) 
circuit.draw(output='mpl')
plt.show()

circuit.measure(qr, cr)
circuit.draw(output='mpl')
plt.show()

simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=simulator).result()
print(result)
plot_histogram(result.get_counts(circuit))