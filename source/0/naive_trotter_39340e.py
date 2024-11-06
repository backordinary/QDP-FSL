# https://github.com/Zshan0/trotterization/blob/a6060cae66459cf00f2ffab5cc2484305df02325/src/naive_trotter.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import I, Z, X, Y

from parser import parse

pauli_gates = [I, X, Y, Z]


def trotter(A: np.ndarray, _time: float, _r: int) -> QuantumCircuit:

    num_qubits = len(A[0])
    qubits = list(range(num_qubits))
    circ = QuantumCircuit(num_qubits, num_qubits)

    for _ in range(_r):
        for term in A:
            operator = pauli_gates[int(term[0])]

            for gate in term[1:]:
                operator = pauli_gates[int(gate)] ^ operator

            evolution_gate = PauliEvolutionGate(operator, _time / _r)
            circ.append(evolution_gate, qubits)

    return circ


def main():
    string = """3 3 10
    x_0 y_1
    x_1 x_2
    z_0"""
    A = parse(string)
    circ = trotter(A, 1.0, 2)
    print(circ)


if __name__ == "__main__":
    main()
