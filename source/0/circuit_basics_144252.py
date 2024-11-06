# https://github.com/twoadave/dgayowsky-code-snippets/blob/d9d8323fe5eaa584789906a671b74cf8da163455/qiskit-tutorials/circuit_basics.py
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:32:26 2022

@author: David

Attempt #2 at creating qiskit tutorial...
"""

#######################################################################

import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Create a Quantum Circuit acting on a quantum register (circuit) of three qubits.
circ = QuantumCircuit(3)

# Set the intial state of the simulator to the ground state using from_int
state = Statevector.from_int(0, 2**3)
#print(state)

# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(0)
#circ.draw(output='mpl')
#plt.show()
#state = state.evolve(circ)
#state.draw('qsphere')
#plt.show()

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
#circ.draw(output='mpl')
#plt.show()
#state = state.evolve(circ)
#state.draw('qsphere')
#plt.show()

# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)
#circ.draw(output='mpl')
#plt.show()
#state = state.evolve(circ)
#state.draw('qsphere')
#plt.show()

#########################################################################

#Create a secondary quantum circuit which takes measurements.
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
# map the quantum measurement to the classical bits
meas.measure(range(3), range(3))

# The Qiskit circuit object supports composition.
# Here the meas has to be first and front=True (putting it before)
# as compose must put a smaller circuit into a larger one.
qc = meas.compose(circ, range(3), front=True)

#drawing the circuit
#qc.draw('mpl')
#plt.show()

backend = AerSimulator()

# First we have to transpile the quantum circuit
# to the low-level QASM instructions used by the
# backend
qc_compiled = transpile(qc, backend)

# Execute the circuit on the qasm simulator.
# Default number of 'shots' is 1024.
job_sim = backend.run(qc_compiled, shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()
counts = result_sim.get_counts(qc_compiled)
print(counts)

plot_histogram(counts)
plt.show()