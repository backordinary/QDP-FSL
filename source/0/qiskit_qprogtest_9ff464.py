# https://github.com/OFThomas/qprogramming/blob/56a52f2d90ec5ebe8ab7b791c9d7402004c0984a/programs/asm/qiskit_qprogtest.py
#!/usr/bin/env python3
from qiskit import ClassicalRegister, QuantumRegister, QuantumProgram
from qiskit import available_backends, execute

# Q in binary
bitstring = '01010001'

# Allocate memory
q = QuantumRegister(8)
c = ClassicalRegister(8)
qprog = QuantumProgram()

qprog.create_classical_register(8)
