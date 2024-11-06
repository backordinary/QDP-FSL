# https://github.com/Eshan-Yadav/quantum-computing-for-string-matching/blob/d32d5db3ed41d2c6520f09211d00259fbc01a34c/document/HGate.py
from matplotlib import backend_bases
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, assemble
from numpy import pi
from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
sim = Aer.get_backend('aer_simulator')

qc = QuantumCircuit(4)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)

print(qc.draw())

qc.save_statevector()
qobj = assemble(qc)
state = sim.run(qobj).result().get_statevector()
plot_bloch_multivector(state).savefig('document\statevectorH.png', dpi=400)


qobj = assemble(qc)  # Assemble circuit into a Qobj that can be run
counts = sim.run(qobj).result().get_counts()  # Do the simulation, returning the state vector
plot_histogram(counts).savefig('document\histogramstatevectorH.png', dpi=1000)