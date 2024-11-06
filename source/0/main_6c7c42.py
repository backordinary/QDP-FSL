# https://github.com/alexrares31/Quantum-Circuits/blob/ec3369231ebda011dddb51c8e1ef63c50760a4f8/Quantum_circuits/Examples/main.py
from qiskit import QuantumCircuit, Aer, assemble
from qiskit.visualization import plot_histogram
import qiskit.quantum_info as qi
import numpy as np

qc = QuantumCircuit(3)
initial_state_0 = [-np.sqrt(3) / 2 * 1.j, 1/2]
initial_state_1 = [np.sqrt(2) / 2 * 1.j, np.sqrt(2) / 2]
qc.initialize(initial_state_0, 0)
qc.initialize(initial_state_1, 1)
qc.y(0)
qc.h(1)
qc.x(2)
qc.z(2)
qc.cx(0, 2)
qc.h(2)
qc.t(2)
qc.cx(2, 1)
qc.h(2)
sim = Aer.get_backend("aer_simulator")
qc.save_statevector()
qobj = assemble(qc)
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts)