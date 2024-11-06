# https://github.com/BOBO1997/unitary_t_design/blob/11b83a72f8137d52b7b8b11bffeb52ba0c20ae5d/qiskit/test_unitaries.py
from typing import Union
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates import Diagonal
import qiskit.quantum_info as qi

def LRC(n: int, D: int, to_gate: bool = False, seeds: list = None) -> QuantumCircuit:
    """
    Local Random Clifford

    Arguments
        n: qubit size
        D: depth
    Returns
        qc: quantum circuit
    """
    qc = QuantumCircuit(n)
    for l in range(1, D+1):
        if n & 1 and l & 1:
            for i in range((n - 1) // 2):
                qc.append(qi.random_unitary(4, seed = seeds[0]).to_instruction(), [2 * i, 2 * i + 1])
        if n & 1 and not l & 1:
            for i in range((n - 1) // 2):
                qc.append(qi.random_unitary(4, seed = seeds[1]).to_instruction(), [2 * i + 1, 2 * i + 2])
        if not n & 1 and l & 1:
            for i in range(n // 2):
                qc.append(qi.random_unitary(4, seed = seeds[2]).to_instruction(), [2 * i, 2 * i + 1])
        if not n & 1 and not l & 1:
            for i in range(n // 2 - 1):
                qc.append(qi.random_unitary(4, seed = seeds[3]).to_instruction(), [2 * i + 1, 2 * i + 2])
        # qc.barrier()

    return qc.to_gate(label="LRC("+str(n)+","+str(D)+")") if to_gate else qc

def RDC(n: int, D: int, to_gate: bool = False, seed: Union[int, np.random.Generator] = None) -> QuantumCircuit:
    """
    Random Diagonal Clifford

    Arguments
        n: qubit size
        D: depth
    Returns
        qc: quantum circuit
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(42)

    qc = QuantumCircuit(n)
    for l in range(D):
        thetas = np.random.uniform(low = 0, high = 2 * np.pi, size = 2 ** n)
        # print(thetas)
        # print(np.exp(thetas * 1j))
        qc.append(Diagonal(np.exp(thetas * 1j)).to_gate(), range(n))
        qc.h(range(n))
    return qc.to_gate(label="RDC("+str(n)+","+str(D)+")") if to_gate else qc
