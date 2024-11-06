# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/Mat3420/controlled_k%20(1).py
import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.circuit.library.standard_gates import SGate, HGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_bloch_multivector



statevector = Statevector.from_label('00')
plot_bloch_multivector(statevector, title='initial state')
plt.show()

qr = qk.QuantumRegister(2)
qc = qk.QuantumCircuit(qr)


h_gate = HGate()
qc.append(h_gate, [qr[1]])

state1 = statevector.evolve(Operator(qc.reverse_bits()))
print(state1)
plot_bloch_multivector(state1, title='Hadamard 1')
plt.show()

cs_gate = SGate().control(1)
qc.append(cs_gate, [qr[0], qr[1]])

state2 = statevector.evolve(Operator(qc.reverse_bits()))
print(state2)
plot_bloch_multivector(state2, title='Controlled K')
plt.show()

qc.append(h_gate, [qr[1]])


state3 = statevector.evolve(Operator(qc.reverse_bits()))
print(state3)
plot_bloch_multivector(state3, title='Hadamard 2')
plt.show()

qc.draw('mpl', reverse_bits='true')

plt.show()

#print(quantum_circuit)
#print(Operator(quantum_circuit))




#state = Statevector.from_instruction(quantum_circuit)
#plot_bloch_multivector(state)
#plt.show()
