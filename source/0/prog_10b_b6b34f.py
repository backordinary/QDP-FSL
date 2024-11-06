# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH03/prog_10b.py
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit 
import math

qc = QuantumCircuit(4) 
qc.rx(math.pi/2, [0,1,2,3]) 
qc.i(1)
qc.u(math.pi/2, 0, math.pi, 2) 
qc.u(0,0, math.pi/4, 3) 
qc.draw("mpl")


state = Statevector.from_instruction(qc) 
state.draw('bloch')