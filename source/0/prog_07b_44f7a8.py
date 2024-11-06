# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH03/prog_07b.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector 

qc = QuantumCircuit(3)
qc.x(0)
qc.y(1)
qc.z(2)
print(qc)
qc.draw("mpl")


state = Statevector.from_instruction(qc) 
state.draw('bloch')