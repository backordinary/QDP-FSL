# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH03/prog_08b.py
from qiskit import QuantumCircuit
import math
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(3)
qc.rx(math.pi/2, 0)
qc.ry(math.pi/2, 1)
qc.rz(math.pi/2, 2)
qc.draw("mpl")


state = Statevector.from_instruction(qc) 
state.draw('bloch')