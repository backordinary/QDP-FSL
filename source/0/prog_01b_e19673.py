# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH02/prog_01b.py
from qiskit import QuantumCircuit
import math
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(4,4)
qc.initialize([1,0],0)
qc.initialize([0,1],1) 
qc.initialize([1/math.sqrt(2), 1/math.sqrt(2)],2) 
qc.initialize([1/math.sqrt(2), -1/math.sqrt(2)],3) 
qc.draw("mpl") 

state = Statevector.from_instruction(qc) 
state.draw('bloch')