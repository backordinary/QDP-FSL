# https://github.com/rili0214/Quantum-Computer-Simulater/blob/99bc0b97a67dccbdd05c76fc5b75438fe6aba6de/CSCI4971_hw6.py
#Team members: Yuming Xie(xiey8), Yuanyi Zhang(zhangy85)

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
import qiskit.quantum_info as qi
from qiskit.circuit.library.standard_gates import (IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate, CXGate, CYGate, CZGate, CHGate, CRZGate, CU1Gate, CU3Gate, SwapGate, RZZGate, CCXGate, CSwapGate)

def f(z, y, x):
    return (x+y)**(x+y+z) + (y+z)**(x+y+z) % 2


def generate_circuit(num_qubits, depth, max_operands=3, measure=False, conditional=False, reset=False, seed=None):
    Gate_type_I = [IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate]
    Gate_type_II = [U1Gate, RXGate, RYGate, RZGate, RZZGate, CU1Gate, CRZGate]
    Gate_type_III = [U2Gate]
    Gate_type_IV = [U3Gate, CU3Gate]
    Gate_type_V = [CXGate, CYGate, CZGate, CHGate, CRZGate, CU1Gate, CU3Gate, SwapGate, RZZGate]
    Gate_type_VI = [CCXGate, CSwapGate]
    Gate_type_VII = QuantumRegister(num_qubits, 'q')
    Gate_type_VIII = QuantumCircuit(num_qubits)
    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        Gate_type_VIII.add_register(cr)
    if reset:
        Gate_type_I += [Reset]
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]
            if num_operands == 1:
                operation = rng.choice(Gate_type_I)
            elif num_operands == 2:
                operation = rng.choice(Gate_type_V)
            elif num_operands == 3:
                operation = rng.choice(Gate_type_VI)
            if operation in Gate_type_II:
                num_angles = 1
            elif operation in Gate_type_III:
                num_angles = 2
            elif operation in Gate_type_IV:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [Gate_type_VII[i] for i in operands]
            op = operation(*angles)
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)
            Gate_type_VIII.append(op, register_operands)
    if measure:
        Gate_type_VIII.measure(Gate_type_VII, cr)
    return Gate_type_VIII

def circuit_to_uninary(circ):
    if __debug__:
        circ = QuantumCircuit(10)
        circ.h(0)
        circ.h(1)
        circ.h(2)
        circ.h(3)
        circ.h(4)
        circ.cx(5,6)
        circ.cx(1,0)
        circ.cx(2,1)
        circ.cx(3,2)
        circ.cx(4,3)
        circ.cx(5,4)
        circ.cx(6,5)
        circ.h(0)
        circ.h(1)
        circ.h(2)
        circ.h(3)
        circ.h(4)
        circ.h(5)
        circ.h(6)
        circ.h(7)
        circ.h(8)
        circ.h(9)
    operator = qi.Operator(circ).data
    backend = Aer.get_backend('unitary_simulator')
    job = backend.run(circ)
    result = job.result()    
    return result.get_unitary(circ, decimals=3)

def compute_output(mtx1, mtx2, flag = True):
    if flag:
        return np.dot(mtx1,mtx2)
    else:
        return np.dot(mtx2,mtx1)
    
def find_index(states):
    vec = "0000000000"
    for k in states:
        vec= vec[:k-1] + "1" + vec[k:]
    return int(vec, 2)

if __name__ == "__main__":
    
    num_qubits = 10; num_levels = 3; control = ([0,0,0,0,0,0,0,0,0,0]); states = [2,4,6,8,10]
    circuit = generate_circuit(num_qubits, num_levels)
    operator = circuit_to_uninary(circuit)
    sol = compute_output(control, operator)
    index = find_index(states)
    with open('Probabilities.txt', 'w') as f:
        f.write("     x          probability\n")
        for i in range(32):
            tmp = format(i, '#050')
            f.write("{}:\t{}\n".format(tmp,str(sol[i])))
        f.write("The possibilities for {} is {:.12s}".format(states,sol))
