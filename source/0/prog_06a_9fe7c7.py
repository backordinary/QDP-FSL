# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH04/prog_06a.py
from qiskit import QuantumCircuit 
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2, 2)
sv = Statevector.from_label("10")
qc.initialize(sv, range(2))
qc.h(0)
qc.cx(0, 1)
qc.measure(range(2), range(2))
qc.draw("mpl")