# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/2ee318932a9c8b395993e10dfe4d9f2cb1d52c28/qiskit/02_circuit_basics.py
import numpy as np
from qiskit import QuantumCircuit

# Create a quantum circuit acting on a quatum registor of three qubits
circ = QuantumCircuit(3)

# By default, each qubit in the register is initialized to |0>.
# To make a GHZ state, we apply the following gates:
# - Hadamard gate H on qubit 0
# - CNOT (Cx) between qubit 0 and qubit 1
# - CNOT (Cx) between qubit 0 and qubit 2

# Hadamard gate on qubit 0 -> superposition
circ.h(0)

# CNOT gate on control qubit 0 and target qubit 1 -> Bell state
circ.cx(0, 1)

# CNOT gate on control qubit 0 and target cubit 2 -> GHZ state
circ.cx(0, 2)

import matplotlib.pyplot as plt
circ.draw("mpl")
plt.savefig("IMG_circuit_01.pdf")

# Simulating circuit
from qiskit.quantum_info import Statevector

# set the input state
state = Statevector.from_int(0, 2**3)

# evolve the state by quantum circuit
state = state.evolve(circ)

state.draw("latex") # for Jupyter notebook

plt.clf()
state.draw("qsphere")