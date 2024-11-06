# https://github.com/VashuKochar/QiskitProjects/blob/c16ba4e4e0832f4ba5678e3ee7899163b27fd7a1/Basics/measurements.py
"""
Measurements
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

# Create quantum circuit with 3 qubits and 3 classical bits:
qc = QuantumCircuit(3, 3)
qc.x([0,1])  # Perform X-gates on qubits 0 & 1
qc.measure([0,1,2], [0,1,2])
qc.draw()    # returns a drawing of the circuit

provider = IBMQ.load_account()
device = Aer.get_backend('aer_simulator')

job = device.run(qc)      # run the experiment
result = job.result()  # get the results
counts= result.get_counts()    # interpret the results as a "counts" dictionary

print(counts)