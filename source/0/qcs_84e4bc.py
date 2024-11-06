# https://github.com/SaraM92/Quantum-Circuit-Slicer/blob/0f2d82338a6ff90b6fd152316f6f1829e679a932/qcs/qcs.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from qiskit.circuit.barrier import Barrier
from qcs.gatefinder import startCount
from qcs.breakbarrier import breakbarrier


def startDebug():
	from qiskit import QuantumCircuit
	setattr(QuantumCircuit, 'breakbarrier', breakbarrier)
	QuantumCircuit = startCount(QuantumCircuit)
	return QuantumCircuit
'''

def startDebug(qc):
	from breakbarrier import breakbarrier
#	from qiskit import QuantumCircuit
	setattr(qc, 'breakbarrier', breakbarrier)
	qc = startCount(qc)
	return qc
#    return QuantumCircuit
'''
