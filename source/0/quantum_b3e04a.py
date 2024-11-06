# https://github.com/mattvdev13/quantum/blob/3f848ca611bab6b8c45140e1802ccb3564cbb5a9/quantum.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np

qc = QuantumCircuit(2)

qc.h(0)

qc.cx(0,1)

qc.measure_all()

qc.draw()

#U = Operator(qc)

# Show the results
#U.data
