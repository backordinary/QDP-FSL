# https://github.com/calwoo/qlearn/blob/8ee23033baf69091c944132378f718ea6c4ce950/algos/teleportation.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, assemble, transpile
from qiskit.visualization import plot_histogram


qr = QuantumRegister(3, name="q")
crz, crx = ClassicalRegister(1, name="crz"), ClassicalRegister(1, name="crx")
teleport_circuit = QuantumCircuit(qr, crz, crx)

# create entangled qubits
def bell_pair(qc, a, b):
    qc.h(a)
    qc.cx(a, b)

bell_pair(teleport_circuit, 1, 2)

# qubits 1 and 2 are entangled qubits
# alice has her secret qubit 0 as well
def alice_gates(qc, a_secret, a_bell):
    qc.cx(a_secret, a_bell)
    qc.h(a_secret)

teleport_circuit.barrier()
alice_gates(teleport_circuit, 0, 1)
teleport_circuit.barrier()

# alice then measures to get classical bits to send to bob
def alice_measure(qc, a_secret, a_bell):
    qc.measure(a_secret, 0)
    qc.measure(a_bell, 1)

alice_measure(teleport_circuit, 0, 1)

# bob then uses classical bits to apply gates
def bob_gates(qc, bob_q, crz, crx):
    qc.x(bob_q).c_if(crx, 1)
    qc.z(bob_q).c_if(crz, 1)

teleport_circuit.barrier()
bob_gates(teleport_circuit, 2, crz, crx)
teleport_circuit.barrier()

teleport_circuit.draw(output="mpl")
plt.show()
