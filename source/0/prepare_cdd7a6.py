# https://github.com/Quantum-Computing-Cooperation/Quantum_Hackathon_2021/blob/39344175b3f5bbf54df3184b0aaa4a11a4cbf245/Guided_exercises/NESS/prepare.py
import qiskit
from qiskit import QuantumCircuit

def prepare(cir):
    tqubits = 3
    q1 = tqubits
    q2 = tqubits+1
    q3 = tqubits+2
    cir.h(q1)
    cir.ch(q1,q2)
    cir.cx(q2,q3)
    cir.x(q1)
    return cir