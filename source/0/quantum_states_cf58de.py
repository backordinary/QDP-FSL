# https://github.com/StevenSchuerstedt/QuantumComputing/blob/0b32d1c642450aec87b2fa0204ba285136875e6b/code/quantum_states.py
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import numpy as np


## statevectors


sim = Aer.get_backend('statevector_simulator')  # Tell Qiskit how to simulate our circuit
#sim = Aer.get_backend('aer_simulator')


qc = QuantumCircuit(2)  # Create a quantum circuit with one qubit
#initial_state = [sqrt(1/2), sqrt(1/4) + sqrt(1/4)*1.j]   # Define initial_state as |1>

initial_state = [1, 0]   # Define initial_state as |0>

qc.initialize(initial_state, 0) # Apply initialisation operation to the 0th qubit

qc.h(0)
qc.x(1)
qc.h(1)


#qc.cx(0, 1)
#qc.y(0)

#meaure collapses statevector
#qc.measure_all()

#qc.save_statevector()   # Tell simulator to save statevector


result = sim.run(qc).result() # Do the simulation and return the result


out_state = result.get_statevector(qc, decimals=3)

print(qc)
print(out_state) # Display the output state vector
plot_bloch_multivector(Statevector.from_instruction(qc), title="New Bloch Multivector", reverse_bits=False)
#print(result.get_counts())


plt.show()