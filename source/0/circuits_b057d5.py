# https://github.com/pablogalaviz/qiskit_demo/blob/a87c49ce52695cedbd96a7f899f50b90d7b1813e/src/circuits.py
import matplotlib.pyplot as plt
from numpy import pi
from qiskit import *
from qiskit.quantum_info import Pauli
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_state_paulivec
from qiskit.circuit import Gate

num_qubits = 2
gate = Gate(name='gate', num_qubits=num_qubits, params=[])

quantum_register = QuantumRegister(num_qubits+1, 'q')
circuit = QuantumCircuit(quantum_register)

circuit.append(gate, [quantum_register[0], quantum_register[1]])
circuit.append(gate, [quantum_register[1], quantum_register[2]])

circuit.draw('mpl')
plt.show()

# %%
# Build a sub-circuit

sub_quantum_register = QuantumRegister(num_qubits)
sub_circuit = QuantumCircuit(sub_quantum_register, name='sub_circ')
sub_circuit.h(sub_quantum_register[0])
sub_circuit.crz(1, sub_quantum_register[0], sub_quantum_register[1])
sub_circuit.barrier()
sub_circuit.id(sub_quantum_register[1])
sub_circuit.u(1, 2, -2, sub_quantum_register[0])

# Convert to a gate and stick it into an arbitrary place in the bigger circuit
sub_instruction = sub_circuit.to_instruction()

circuit.append(sub_instruction, [quantum_register[1], quantum_register[2]])
circuit.draw('mpl')
plt.show()

# %%

decomposed_circuit = circuit.decompose() # Does not modify original circuit
decomposed_circuit.draw('mpl')
plt.show()
