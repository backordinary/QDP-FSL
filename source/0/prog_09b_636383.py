# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH03/prog_09b.py
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit 
import math

qc = QuantumCircuit(4) 
qc.rx(math.pi/2, [0,1,2,3]) 
qc.p(math.pi/8, 1)
qc.s(2)
qc.t(3)
qc.draw("mpl")


state = Statevector.from_instruction(qc) 
state.draw('bloch')