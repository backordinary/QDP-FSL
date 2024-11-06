# https://github.com/t11s/NEQR-examples/blob/0bd3482451b328d8419c5209d10ab66393c9ae07/NEQR_example_4_pixel_8bpp.py
# This script sets up a quantum circuit to load an NEQR quantum image with four pixels with 8-bit grayscale values
#
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(10, 'q')
creg_c = ClassicalRegister(10, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.reset(qreg_q[1])
circuit.reset(qreg_q[2])
circuit.reset(qreg_q[3])
circuit.reset(qreg_q[4])
circuit.reset(qreg_q[5])
circuit.reset(qreg_q[6])
circuit.reset(qreg_q[7])
circuit.reset(qreg_q[8])
circuit.reset(qreg_q[9])
circuit.h(qreg_q[8])
circuit.h(qreg_q[9])
circuit.cx(qreg_q[9], qreg_q[0])
circuit.x(qreg_q[9])
circuit.ccx(qreg_q[8], qreg_q[9], qreg_q[1])
circuit.x(qreg_q[9])
circuit.cx(qreg_q[9], qreg_q[1])
circuit.cx(qreg_q[8], qreg_q[2])
circuit.ccx(qreg_q[8], qreg_q[9], qreg_q[3])
circuit.cx(qreg_q[9], qreg_q[4])
circuit.cx(qreg_q[8], qreg_q[5])
circuit.ccx(qreg_q[8], qreg_q[9], qreg_q[6])
circuit.ccx(qreg_q[8], qreg_q[9], qreg_q[7])
#
# Measure
#
circuit.measure(qreg_q[0], creg_c[9])
circuit.measure(qreg_q[1], creg_c[8])
circuit.measure(qreg_q[2], creg_c[7])
circuit.measure(qreg_q[3], creg_c[6])
circuit.measure(qreg_q[4], creg_c[5])
circuit.measure(qreg_q[5], creg_c[4])
circuit.measure(qreg_q[6], creg_c[3])
circuit.measure(qreg_q[7], creg_c[2])
circuit.measure(qreg_q[9], creg_c[1])
circuit.measure(qreg_q[8], creg_c[0])
