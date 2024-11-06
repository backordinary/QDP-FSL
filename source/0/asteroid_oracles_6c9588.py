# https://github.com/pdebus/IBMQuantumChallenge2020/blob/366552f9f8129f73d5483f8c9d7f9df1afb8d929/src/asteroid_oracles.py
from qiskit import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute

import numpy as np
import matplotlib.pyplot as plt

from utils.board_tools import kbits, compute_uncovered_tiles, board2_bitstrings
from circuit_parts.diffusers import diffuser_gate, diffuser
from board_qram import init_light_states

from datasets.data import *


def single_beam_check(beamstring, qc, board_qubits, output_qubit, ancilla_qubits=None):

    qubit_idx = compute_uncovered_tiles(beamstring)
    num_controls = len(qubit_idx)
    num_ancillas = num_controls - 2

    control_qubits = [board_qubits[i] for i in qubit_idx]

    if ancilla_qubits is not None and num_ancillas > 0:
        qc.mct(control_qubits, output_qubit, ancilla_qubits[:num_ancillas], mode='basic')
    else:
        qc.mct(control_qubits, output_qubit)


def beam_checker(qc, board_qubits, output_qubit, ancilla_qubits=None):

    board_size = len(board_qubits)
    board_dim = int(np.sqrt(board_size))

    beamstrings = kbits(2 * board_dim, board_dim - 1)

    for b in beamstrings:
        single_beam_check(b, qc, board_qubits, output_qubit, ancilla_qubits)


def single_board_checker(board, draw=False):

    board_size = len(board)

    board_qubits = QuantumRegister(board_size, name='board')
    output_qubit = QuantumRegister(1, name='out')
    ancilla_qubits = QuantumRegister(7, name='ancilla')

    cbits = ClassicalRegister(1, name="cbits")

    qc = QuantumCircuit(board_qubits, ancilla_qubits, output_qubit, cbits)
    init_light_states(qc, board, board_qubits)

    beam_checker(qc, board_qubits, output_qubit, ancilla_qubits)

    qc.measure(output_qubit, cbits)

    if draw:
        qc.draw(output='mpl')
        plt.show()

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=100)
    result = job.result()
    count = result.get_counts()

    max_k = min(len(count), 10)

    for k in sorted(count, key=count.get, reverse=True)[:max_k]:
        print(f"Solution: {k}\tCounts: {count[k]}")


if __name__ == "__main__":
    # single_board_checker([1, 0, 0, 0, 1, 0, 0, 0, 1], draw=True)
    # single_board_checker([1, 0, 0, 1, 1, 1, 0, 0, 0], draw=True)
    # single_board_checker([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0], draw=True)
    # single_board_checker([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], draw=True)

    # for n, problem in enumerate(board2_bitstrings(problem_set3x3, board_size=3)):
    for n, problem in enumerate(board2_bitstrings(q3)):
        print(f"Board {n}:")
        single_board_checker(problem)
