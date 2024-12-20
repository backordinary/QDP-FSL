# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH04/prog_04a.py
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(8, 8)
sv = Statevector.from_label("11011000")
qc.initialize(sv, range(8))
qc.cx(0, 1)
qc.cx(2, 3)
qc.cx(4, 5)
qc.cx(6, 7)
qc.measure(range(8), range(8))

qc.draw("mpl")