# https://github.com/jongjinna/Quantum_IBM/blob/6df03de5de8fefc014c44b6e630ca0e3a6d6b657/IBM_Ex/IBM_2_6.py
from qiskit import QuantumCircuit, QuantumRegister

input_bit = QuantumRegister(1, 'input')
output_bit = QuantumRegister(1, 'output')
garbage_bit = QuantumRegister(1, 'garbage')

Uf = QuantumCircuit(input_bit, output_bit, garbage_bit)
Uf.cx(input_bit[0], output_bit[0])
print(Uf.draw())

Vf = QuantumCircuit(input_bit, output_bit, garbage_bit)
Vf.cx(input_bit[0], garbage_bit[0])
Vf.cx(input_bit[0], output_bit[0])
print(Vf.draw())

qc = Uf + Vf.inverse()
print(qc.draw())

final_output_bit = QuantumRegister(1, 'final-output')

copy = QuantumCircuit(output_bit, final_output_bit)
copy.cx(output_bit, final_output_bit)

print(copy.draw())

print((Vf.inverse() + copy + Vf).draw())
