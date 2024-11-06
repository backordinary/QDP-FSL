# https://github.com/VashuKochar/QiskitProjects/blob/c16ba4e4e0832f4ba5678e3ee7899163b27fd7a1/Basics/halfAdder.py
"""
Half Adder
"""

import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, assemble
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.visualization import *
from ibm_quantum_widgets import *
import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector,array_to_latex
from qiskit.providers.ibmq import IBMQ, least_busy

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)

qc.h(1)
qc.cx(1,0)

ket = Statevector(qc)
print(ket)