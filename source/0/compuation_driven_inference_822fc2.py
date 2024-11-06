# https://github.com/Eshan-Yadav/quantum-computing-for-string-matching/blob/d32d5db3ed41d2c6520f09211d00259fbc01a34c/document/Compuation_driven_inference.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
import numpy as np
from qiskit import Aer
from qiskit.visualization import plot_histogram
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.circuit.library import PhaseOracle
from matplotlib import backend_bases
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, assemble
from numpy import pi
from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
sim = Aer.get_backend('aer_simulator')

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.reset(qreg_q[1])
circuit.reset(qreg_q[2])
circuit.ry(1.9106332362490184, qreg_q[0])
circuit.cu(pi / 2, pi / 2, pi / 2, pi / 2, qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.x(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
print(circuit.draw())


circuit.save_statevector()
qobj = assemble(circuit)
state = sim.run(qobj).result().get_statevector()
plot_bloch_multivector(state).savefig('document\computation_driven.png', dpi=400)


qobj = assemble(circuit)  # Assemble circuit into a Qobj that can be run
counts = sim.run(qobj).result().get_counts()  # Do the simulation, returning the state vector
plot_histogram(counts).savefig('document\computation_driven_Vec.png', dpi=400)
plot_histogram(circuit.measure_counts(shots=1024))



