# https://github.com/ArfatSalman/qc-test/blob/9ec9efff192318b71e8cd06a49abc676196315cb/miscellanea/additional_warnings/12_4c1fb6/source_4c1fb6217e2f4e359a721e68ddc7077e.py

# SECTION
# NAME: PROLOGUE

import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library.standard_gates import *
from qiskit.circuit import Parameter

# SECTION
# NAME: CIRCUIT

qr = QuantumRegister(5, name='qr')
cr = ClassicalRegister(5, name='cr')
qc = QuantumCircuit(qr, cr, name='qc')

# SECTION
# NAME: MEASUREMENT

qc.measure(qr, cr)

# SECTION
# NAME: OPTIMIZATION_LEVEL

from qiskit import transpile
qc = transpile(qc, basis_gates=None, optimization_level=0, coupling_map=None)

# SECTION
# NAME: EXECUTION

from qiskit import Aer, transpile, execute
backend_04ca3b30d0b44cff8ee7c7aabe5dd837 = Aer.get_backend('qasm_simulator')
counts = execute(qc, backend=backend_04ca3b30d0b44cff8ee7c7aabe5dd837, shots=979).result().get_counts(qc)
RESULT = counts