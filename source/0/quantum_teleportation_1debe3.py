# https://github.com/Thakkar-meet/Quantum-Computing/blob/1b921565abe4993cd8645531e1d7030f1b19f3bb/quantum_teleportation.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def draw(qc):
    qc.draw(output='mpl')
    plt.show()


def create_bell_pair(qc,a,b):
    qc.h(a)
    qc.cx(a,b)
    return qc


def alice_gates(qc, psi, a):
    qc.cx(psi,a)
    qc.h(psi)


def measure_and_send(qc,a,b):
    qc.barrier()
    qc.measure(a,0)
    qc.measure(b,1)
    return qc


def bob_gates(qc, crx, crz, qubit):
    qc.x(qubit).c_if(crx,1)
    qc.z(qubit).c_if(crz,1)
    return qc


qr = QuantumRegister(3,"q")
crz = ClassicalRegister(1, name="crz")
crx = ClassicalRegister(1, name="crx")
teleportation_circuit = QuantumCircuit(qr,crz,crx)

create_bell_pair(teleportation_circuit,1,2)
teleportation_circuit.barrier()

alice_gates(teleportation_circuit,0,1)

measure_and_send(teleportation_circuit,0,1)
teleportation_circuit.barrier()

bob_gates(teleportation_circuit, crx, crz, 2)
draw(teleportation_circuit)


