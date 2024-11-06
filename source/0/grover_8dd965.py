# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/grover.py
from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import QuantumCircuit

# Criando o circuito oracle #

oracle = QuantumCircuit(2, name='Oráculo')
oracle.cz(0,1)
oracle.to_gate()

backend = Aer.get_backend('statevector_simulator')
grover_circuit = QuantumCircuit(2,2)
grover_circuit.h([0,1])
grover_circuit.append(oracle, [0,1])

job = execute(grover_circuit, backend)
result = job.result()

sv = result.get_statevector()
print(np.around(sv,2))

reflection = QuantumCircuit(2, name ='Grover')
reflection.h([0,1])
reflection.z([0,1])
reflection.cz(0,1)
reflection.h([0,1])
reflection.to_gate()

backend = Aer.get_backend('qasm_simulator')
grover_circuit = QuantumCircuit(2,2)
grover_circuit.h([0,1])
grover_circuit.append(oracle, [0,1])
grover_circuit.append(reflection, [0,1])
grover_circuit.measure([0,1],[0,1])
grover_circuit.draw(output='mpl')
reflection.draw(output='mpl')
plt.show()

job = execute(grover_circuit, backend, shots = 1)
result = job.result()
print(result.get_counts())