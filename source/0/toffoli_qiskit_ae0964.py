# https://github.com/OFThomas/qprogramming/blob/56a52f2d90ec5ebe8ab7b791c9d7402004c0984a/report/code/QISKit/toffoli_qiskit.py
# Implementation of Control-Control-NOT or Toffoli in Qiskit

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, QuantumProgram
from qiskit import available_backends, execute
import numpy as np


# Initialise 3 qubit register and classical readout
q = QuantumRegister(3, 'ctrl')
c = ClassicalRegister(3, 'meas')

# Combine resources into a quantum circuit
qc = QuantumCircuit(q, c)

qc.ccx(q[0],q[1],q[2])
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])

# Execute the quantum circuit on the local simulator
job = execute(qc, 'local_qasm_simulator')
result = job.result()
print('The results of the simulation shots are:', result.get_counts(qc))

