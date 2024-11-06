# https://github.com/prthmsh77/qtgame/blob/db25aaad9b6a9bc466f950b30fac5f7f1083abeb/qtgame/entgld.py
from qiskit import QuantumCircuit, Aer
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

def entangled():
    qc = QuantumCircuit(2,2)
    # Apply H-gate to the first:
    qc.h(0)
    # Apply a CNOT:
    qc.cx(0,1)
    qc.measure([0,1],[0,1])
    qsim = Aer.get_backend('aer_simulator')
    result = qsim.run(qc).result().get_counts()
    return result
def img():
    qc = QuantumCircuit(2)
    # Apply H-gate to the first:
    qc.h(0)
    # Apply a CNOT:
    qc.h(1)
    state = Statevector.from_instruction(qc)
    img = plot_bloch_multivector(state)
    return img

