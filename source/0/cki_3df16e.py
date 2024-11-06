# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/Mat3420/cki.py
import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import Operator



circ = qk.QuantumCircuit(3)
circ.crz(-np.pi/2, 0, 1)

print(Operator(circ))

print(circ)
circ.draw('mpl')
plt.show()
