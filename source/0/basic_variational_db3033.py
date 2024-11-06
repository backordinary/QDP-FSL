# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/Fys4411/src/circuitDiagrams/basic_variational.py
import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.circuit import Gate, Parameter


q_reg = qk.QuantumRegister(3)
c_reg = qk.ClassicalRegister(1)
circuit = qk.QuantumCircuit(q_reg, c_reg)

theta = Parameter('θ')


U_enc = Gate(name='U', num_qubits=3, params=[])
U_anz = Gate(name='U', num_qubits=3, params=[])

circuit.append(U_enc, [q_reg[0], q_reg[1], q_reg[2]])
circuit.append(U_anz, [q_reg[0], q_reg[1], q_reg[2]])
circuit.measure(q_reg[2], c_reg[0])



print(circuit.draw(output='latex_source'))

circuit.draw(output='mpl')
plt.show()
