# https://github.com/Advanced-Research-Centre/QPULBA/blob/42ecc9c6e556fe0e5cb7b352f863271365969f02/archived/WinCondaQiskitTest/try004.py
from qiskit import QuantumCircuit
circ = QuantumCircuit(1)

import numpy as np
# circ.initialize([1, 1, 0, 0] / np.sqrt(2), [0, 1])
circ.ry(np.pi/2,0)

from qiskit import Aer, execute
simulator = Aer.get_backend('statevector_simulator')
sim_res = execute(circ, simulator).result()
statevector = sim_res.get_statevector(circ)
for i in statevector:
	print(i)