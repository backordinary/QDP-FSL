# https://github.com/Algorithmist-Girl/QuantumComputingConcepts_From_IBMTextbook/blob/de7810f9aa8c2a30fb0f1178e67d3be406069538/QuantumStates&Qubits/SingleQubitGates.py
from qiskit import *
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

from math import  pi
#gates = operations that changes qubits!

# Pauli gate!!
# multiply qubit's statevector by the gate to see the effect!!

# 1 qubit!!
quantum_circuit = QuantumCircuit(1)
quantum_circuit.x(0)
quantum_circuit.draw('mpl')
plt.show()

# plot_bloch_multivector takes in the qubit's statevector instead of the Bloch vector!!
backend = Aer.get_backend('statevector_simulator')
res= execute(quantum_circuit, backend).result().get_statevector()
plot_bloch_multivector(res)
plt.show()

# Pauli Y and Z gates ==> rotations about the y and z axes!!
quantum_circuit.y(0)
quantum_circuit.z(0)
plt.show()

#  but these gates operate very similar to classical bits ==> want to instead use superposition!!
#  use Hadamard gate instead!
#  superposition of |0> and |1> ==> hadamard gate
quantum_circuit.h(0)
quantum_circuit.draw('mpl')
plt.show()
