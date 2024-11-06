# https://github.com/harry1357/Quantum/blob/8a6f567a2d872fdca2663c6d64434c56c4fd7c92/Swap%20to%20cnot%20gate.py
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *

from qiskit.visualization import *

from ibm_quantum_widgets import *

from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import *

from qiskit.visualization import plot_state_city

import numpy as np

provider = IBMQ.load_account()

qc = QuantumCircuit(2)

qc.swap(0,1)

qc.draw()
