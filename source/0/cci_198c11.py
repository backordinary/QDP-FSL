# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/Mat3420/cci.py
import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import Operator



circ = qk.QuantumCircuit(3)
circ.h(2)                                                                       # applying the HADAMARD gate to the third qubit
circ.ccx(0, 1, 2)                                                               # applying the CC NOT (toffoli) gate to the third qubit with 1st and 2nd as reference
circ.h(2)
circ.sdg(2)                                                                     # applying the CONJUGATE PHASE GATE (s^-1) to the third cubit
circ.ccx(0, 1, 2)
circ.s(2)                                                                       # applying the PHASE GATE to the third qubit
circ.ccx(0, 1, 2)


print(Operator(circ))

print(circ)
circ.draw('mpl')
plt.show()
