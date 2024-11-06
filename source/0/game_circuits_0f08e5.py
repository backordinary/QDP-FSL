# https://github.com/Jabi7/Internship-2022-summer/blob/6520965aa623e6a6450a0202285a8ec0ea00138f/Belief-Invariant%20and%20Quantum%20Equilibriain%20Games%20of%20Incomplete%20Information/game_circuits.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, assemble, Aer, visualization
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
pi = np.pi

simulator = Aer.get_backend('qasm_simulator')

## parameterized initial state for 2-qubit
def psi(circuit, qreg_q, y1 = pi/2, z1 = 0, y2 = 0, z2 = 0):
    circuit.ry(y1, qreg_q[0])
    circuit.ry(y2, qreg_q[1])
    circuit.rz(z1, qreg_q[0])
    circuit.rz(z2, qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    
## parameterized initial state for 3-qubit
def psi3(circuit, qreg_q, y1 = pi/2, z1 = 0, y2 = -pi/2, z2 = -pi, y3 = -pi, z3 = pi): 
    circuit.ry(y1, qreg_q[0])
    circuit.ry(y2, qreg_q[1])
    circuit.ry(y3 / 2, qreg_q[2])

    circuit.cx(qreg_q[0], qreg_q[2])
    
    circuit.rz(z1, qreg_q[2])
    circuit.rz(z2, qreg_q[1])
    circuit.rz(z3, qreg_q[0])
    circuit.cz(qreg_q[0], qreg_q[2])
    circuit.cz(qreg_q[0], qreg_q[1])
    circuit.cz(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[2])

## parameterized measurement strategy    
def M(circuit, q, c, y = 0, z = 0):
    circuit.ry(y, q)
    circuit.rz(z, q)
    circuit.measure(q, c)
    
def probds(qb, shots =1000):
    job = execute(qb, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts(qb)
    c = dict(sorted(counts.items()))
    p = [i/shots for i in list(c.values())]
    plt.bar(list(c.keys()), p, color='g')
    plt.show()