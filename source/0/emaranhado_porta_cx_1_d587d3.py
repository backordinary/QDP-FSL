# https://github.com/rodrigopff/ibm_quantica/blob/1af9ce3a71bc3949d422dad6f7aecefcb225a547/emaranhado_PORTA_CX_1.py
"""
Codigo para o estudo de emaramnamento quantico do tutorial da IBM:
Portas CX sem emaranhamento. (primeiro circuitinho)
https://learn.qiskit.org/course/introduction/entangled-states

"""

from qiskit import QuantumCircuit
#import qiskit.circuit.quantumcircuit as qq # para importar o modulo todo 

import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

# somente testes se quiser usar o modulo quantumcircuit todo
# q = qq.QuantumCircuit(2)
# q.draw('mpl')
# plt.show()

qc = QuantumCircuit(2)
# qc.x(0)
# qc.cx(0,1)
qc.x(1)
qc.cx(1,0)
ket = Statevector(qc)
print(ket.draw())
qc.draw('mpl')
plt.show()