# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/from_label.py
import qiskit
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city
from qiskit.circuit.random import random_circuit
from matplotlib import pyplot as plt
# Crate a Random Circuit
qc = QuantumCircuit(2,4)

sv = Statevector.from_label('01')
ev = sv.evolve(qc) # Now inject state vector to QC Circuit.
# We can plot using either method!
plot_state_city(ev)
ev.draw('city')
plt.show()