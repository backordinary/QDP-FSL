# https://github.com/G-Carneiro/GCQ/blob/a557c193c54c4f193c9ffde7f94c576b06972abe/src/grover/grover.py
from math import sqrt, pi
from typing import List

from qiskit import QuantumCircuit, execute
from qiskit.circuit.quantumregister import Qubit
from qiskit.providers.aer.backends.aer_simulator import AerSimulator


def phase_oracle(qc: QuantumCircuit, state: int) -> None:
    state: str = bin(state)[2:]
    state = "0" * (qc.num_qubits - len(state)) + state
    flip_qubits: List[Qubit] = []
    state = state[::-1]
    for i in range(len(state)):
        if (state[i] == "0"):
            flip_qubits.append(qc.qubits[i])

    if flip_qubits:
        qc.x(flip_qubits)
    qc.h(qc.qubits[-1])
    qc.mcx(qc.qubits[:-1], qc.qubits[-1])
    qc.h(qc.qubits[-1])
    if flip_qubits:
        qc.x(flip_qubits)

    return None


def grover_diffuser(qc: QuantumCircuit) -> None:
    qc.h(qc.qubits)
    qc.x(qc.qubits)

    # Make a multi controlled z gate
    qc.h(qc.num_qubits - 1)
    qc.mcx(qc.qubits[:-1], qc.num_qubits - 1)
    qc.h(qc.num_qubits - 1)

    qc.x(qc.qubits)
    qc.h(qc.qubits)

    return None


def grover_operator(qc: QuantumCircuit, states: List[int]) -> None:
    for state in states:
        phase_oracle(qc, state)

    grover_diffuser(qc)

    return None


def grover(states: List[int], num_qubits: int) -> int:
    circuit: QuantumCircuit = QuantumCircuit(num_qubits)

    entries = 2 ** num_qubits
    steps = int((pi / 4) * sqrt(entries / len(states)))

    circuit.h(circuit.qubits)

    for _ in range(steps):
        grover_operator(circuit, states)

    circuit.measure_all()

    sim = AerSimulator()
    counts = execute(circuit, sim, shots=1).result().get_counts()

    return int(list(counts)[0], 2)
