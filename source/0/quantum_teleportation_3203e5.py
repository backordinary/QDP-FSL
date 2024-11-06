# https://github.com/Crabster/qiskit-learning/blob/3f14c39ee294f42e3f83a588910b659280556a68/circuits/quantum_teleportation.py
import qiskit 

from .common_gates import *

def quantum_teleportation_circuit():
    qc = qiskit.QuantumCircuit(3)

    phi_plus = phi_plus_gate()
    qc.append(phi_plus, [1, 2])
    qc.append(phi_plus.inverse(), [0, 1])

    qc.barrier()

    qc.cx(1, 2)
    qc.cz(0, 2)

    qc.name = "TP"
    return qc

def quantum_teleportation_example():
    qc = qiskit.QuantumCircuit(3, 1)

    rand_state = random_state_gate()
    qc.append(rand_state, [0])

    tp_qc = quantum_teleportation_circuit()
    qc.append(tp_qc, [0, 1, 2])

    qc.append(rand_state.inverse(), [2])
    qc.measure([2], [0])

    return qc
