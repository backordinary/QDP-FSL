# https://github.com/BrantNorrisUFL/PreAlphaBuild/blob/c8b38eec77d7cfd69106cdfd565587ee3ecf5a63/Qiskit%20Test%201.py
#!/usr/bin/env python
# coding: utf-8


import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

quanCrct = QuantumCircuit(4, 4)

quanCrct.h(0) # Adding H gate to first qubit (qubit 0). This creates a superposition on this qubit

quanCrct.cx(0,1) # Add a CNOT gate on qubits 0 (control) and 1 (target) creating an entanglement (Bell state)

quanCrct.h(2) # Add H gate to qubit 2, superposition

quanCrct.cx(2,3) # Add CNOT Gate on qubit 2 and 3 (control qubit 2 and target 3)

quanCrct.draw('mpl') # Draw circuit with matplotlib

U = Operator(quanCrct)
U.data

quanCrct.h(2)

quanCrct.h(3)

quanCrct.y(2)

quanCrct.z(3)

quanCrct.draw('mpl') # Draw circuit with matplotlib


U = Operator(quanCrct)
U.data

quanCrct.cx(1,2)

quanCrct.draw('mpl')

U = Operator(quanCrct)
U.data


meas = QuantumCircuit(4, 4) # Create circuit to measure
meas.measure([0,1], [0,1])

backend = BasicAer.get_backend('qasm_simulator') #

circ = quanCrct.compose(meas)

result = backend.run(transpile(circ, backend), shots=1000).result()

counts  = result.get_counts(quanCrct)
print(counts)

plot_histogram(counts)
