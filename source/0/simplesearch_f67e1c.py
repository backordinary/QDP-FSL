# https://github.com/Eshan-Yadav/quantum-computing-for-string-matching/blob/d32d5db3ed41d2c6520f09211d00259fbc01a34c/document/SimpleSearch.py
from matplotlib import backend_bases
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, assemble
from numpy import pi
from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
sim = Aer.get_backend('aer_simulator')

qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[1])
circuit.h(qreg_q[0])
circuit.cz(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.z(qreg_q[0])
circuit.z(qreg_q[1])
circuit.cz(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])

qc=circuit

print(qc.draw())

qc.save_statevector()
qobj = assemble(qc)
state = sim.run(qobj).result().get_statevector()
plot_bloch_multivector(state).savefig('document\GroverStateVector.png', dpi=400)


qobj = assemble(qc)  # Assemble circuit into a Qobj that can be run
counts = sim.run(qobj).result().get_counts()  # Do the simulation, returning the state vector
plot_histogram(counts).savefig('document\GroverHistogram.png', dpi=1000)