# https://github.com/alphakilo11/Python/blob/89a07ae0442eb0234bb89da602d5fad3679bceab/Quantum_Computing/QFT.py
# https://www.heise.de/hintergrund/Quantencomputer-programmieren-Nur-eine-Phase-7358148.html

#!pip install qiskit
#!pip install pylatexenc

import pylatexenc
import qiskit
from numpy import pi

QFTcircuit = qiskit.QuantumCircuit(3, 3)
init_state = [1, 1, 0]
for qubit, state in enumerate(init_state):
  if state == 1:
    QFTcircuit.x(qubit)

QFTcircuit.h(0)
QFTcircuit.cp(pi / 2, 1, 0)
QFTcircuit.cp(pi / 4, 2, 0)

#rekursiv für q1
QFTcircuit.h(1)
QFTcircuit.cp(pi/2, 2, 1)

#rekursiv für q2
QFTcircuit.h(2)

#QFTcircuit.measure([0,1,2], [0, 1, 2])
QFTcircuit.draw(output='mpl') # thros a missing-library error, obwohl pylatexenc installiert und importiert wurde
