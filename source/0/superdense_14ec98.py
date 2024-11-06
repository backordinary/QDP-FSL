# https://github.com/VashuKochar/QiskitProjects/blob/c16ba4e4e0832f4ba5678e3ee7899163b27fd7a1/Basics/superdense.py
"""
Superdense

NOT WORKING
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

qc_charlie = QuantumCircuit(2,2)

qc_charlie.h(1)
qc_charlie.cx(1,0)

MESSAGE = '01'

qc_alice = QuantumCircuit(2,2)

if MESSAGE[-2]=='1':
    qc_alice.z(1)
if MESSAGE[-1]=='1':
    qc_alice.x(1)

provider = IBMQ.load_account()
backend = Aer.get_backend('aer_simulator')

complete_qc = qc_charlie.compose(qc_alice.compose(qc_bob))
counts = backend.run(complete_qc).result().get_counts()
print(counts)