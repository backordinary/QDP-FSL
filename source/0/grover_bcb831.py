# https://github.com/anaisafonseca/computacaoQuantica/blob/97bf461f4db1b2e406b8fa7c4ee46b78df711cab/grover.py
# ANAISA FORTI DA FONSECA
# 11811ECP012

from ast import In
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit, Aer, execute, QuantumRegister, ClassicalRegister
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram

def Initialize(QC, Qubits):
    for q in Qubits:
        QC.h(q)
    return QC

# o código a seguir simula um circuito de grover com 2 qubits
n = 2
grover_circuit = QuantumCircuit(n)
grover_circuit = Initialize(grover_circuit, [0,1])
# grover_circuit.draw(output='mpl')
# plt.show()

# o código a seguir simula uma matriz oráculo com 2 qubits
grover_circuit.cz(0,1)
# grover_circuit.draw(output='mpl')
# plt.show()

# o código a seguir simula um difusor para 2 qubits
grover_circuit.h([0,1])
grover_circuit.z([0,1])
grover_circuit.ch(0,1)
grover_circuit.h([0,1])
# grover_circuit.draw(output='mpl')
# plt.show()

# simulando o circuito...
sv_sim = Aer.get_backend('statevector_simulator')
job_sim = execute(grover_circuit, sv_sim)
statevec = job_sim.result().get_statevector()

# medindo os qubits
grover_circuit.measure_all()

qasm_simulator = Aer.get_backend('qasm_simulator')
shots = 1024  # valor padrão
results = execute(grover_circuit, backend = qasm_simulator, shots = shots).result()
answer = results.get_counts()
plot_histogram(answer)
plt.show()