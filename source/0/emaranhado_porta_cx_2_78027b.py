# https://github.com/rodrigopff/ibm_quantica/blob/a6d3d763134aeaad4ae4a272bebdb4a0693189d8/emaranhado_PORTA_CX_2.py
"""
Codigo para o estudo de emaranhamento quantico do tutorial da IBM:
Portas CX com emaranhamento (segundo circuitinho).
https://learn.qiskit.org/course/introduction/entangled-states

"""
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

# Let's create a fresh quantum circuit
qc = QuantumCircuit(2)
qc.h(1) # aplica porta Hadamard ao segundo qubit (q1)
ket = Statevector(qc)
print(ket.draw())
# qc.draw('mpl')
# plt.show()

# Aplica porta CX (Cnot) no primeiro qubit (q0)
# Nada acontece pois o qubit de controle  é 0 : |00>
# qc.cx(0,1)
# ket = Statevector(qc)
# print(ket.draw())
# qc.draw('mpl')
# plt.show()

# Aplica porta CX (Cnot) no segundo qubit (q1) para emaranhar o estado.
# como o qbit de controle é 1 ele troca o qbit alvo(target) : |10> => |11>
qc.cx(1,0)
ket = Statevector(qc)
#print(ket.draw())
qc.draw('mpl')
# plt.show(block=False)
plt.pause(interval=1.0) # Pltar oc paus antes de terminar