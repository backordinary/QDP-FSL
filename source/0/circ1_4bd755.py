# https://github.com/ironmanaudi/Machine-Learning-for-QEM/blob/30cdf08c6280b452620f94a0532122c3bc656ac5/pic/circ1.py
import qiskit

from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
circuit_drawer(qc, filename='circuit.png')
