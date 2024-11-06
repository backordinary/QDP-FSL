# https://github.com/calwoo/qlearn/blob/8ee23033baf69091c944132378f718ea6c4ce950/qubits/uncomputation.py
from typing import final
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, assemble, Aer
from qiskit.visualization import plot_histogram


input_bit = QuantumRegister(1, "input")
output_bit = QuantumRegister(1, "output")
garbage_bit = QuantumRegister(1, "garbage")
final_output_bit = QuantumRegister(1, "final_output")

qc_v = QuantumCircuit(input_bit, output_bit, garbage_bit)
# qc first copies input to output and garbage bits using CNOT gates
qc_v.cx(input_bit[0], output_bit[0])
qc_v.cx(input_bit[0], garbage_bit[0])

qc_v.draw(output="mpl")
plt.show()

qc_copy = QuantumCircuit(output_bit, final_output_bit)
# copy output bit of qc_v to its own bit
qc_copy.cx(output_bit[0], final_output_bit[0])

(qc_v + qc_copy + qc_v.inverse()).draw(output="mpl")
plt.show()


