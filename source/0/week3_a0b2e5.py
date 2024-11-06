# https://github.com/pdebus/IBMQuantumChallenge2020/blob/366552f9f8129f73d5483f8c9d7f9df1afb8d929/src/week3.py
from qiskit import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute

import numpy as np
import matplotlib.pyplot as plt

from board_qram import write_qram, qRAM
from asteroid_oracles import beam_checker
from circuit_parts.diffusers import diffuser, diffuser_gate
from board_tools import board2_bitstrings
from datasets.data import *


def downsized(boards, num_iterations=1):
    board_qubits = QuantumRegister(9, name='board')
    address_qubits = QuantumRegister(4, name='address')
    ancilla_qubits = QuantumRegister(7, name='ancilla')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(4, name="cbits")
    qc = QuantumCircuit(board_qubits, address_qubits, ancilla_qubits, output_qubit, cbits)

    # Initialization
    # Flag to |->
    qc.x(output_qubit)
    qc.h(output_qubit)

    # Address to |++++>
    qc.h(address_qubits)

    write_qram(boards, qc, address_qubits, board_qubits)
    qc.barrier()

    beam_checker(qc, board_qubits, output_qubit, ancilla_qubits)
    # qc.x([board_qubits[0], board_qubits[4], board_qubits[8]])
    # qc.mct([board_qubits[0], board_qubits[4], board_qubits[8]], output_qubit, ancilla_qubits[:1])
    # qc.x([board_qubits[0], board_qubits[4], board_qubits[8]])

    qc.barrier()

    write_qram(boards, qc, address_qubits, board_qubits)
    qc.barrier()

    diffuser(qc, address_qubits)

    qc.barrier()
    qc.measure(address_qubits, cbits)

    #qc = qc.reverse_bits()

    return qc


def week3_ans_func(boards, num_iterations=1):
    board_qubits = QuantumRegister(16, name='board')
    address_qubits = QuantumRegister(4, name='address')
    ancilla_qubits = QuantumRegister(7, name='ancilla')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(4, name="cbits")
    qc = QuantumCircuit(board_qubits, address_qubits, ancilla_qubits, output_qubit, cbits)

    # Initialization
    # Flag to |->
    qc.x(output_qubit)
    qc.h(output_qubit)

    # Address to |++++>
    qc.h(address_qubits)

    write_qram(boards, qc, address_qubits, board_qubits, ancilla_qubits)
    qc.barrier()

    beam_checker(qc, board_qubits, output_qubit, ancilla_qubits)

    qc.barrier()

    write_qram(boards, qc, address_qubits, board_qubits, ancilla_qubits)
    qc.barrier()

    diffuser(qc, address_qubits)

    qc.barrier()
    qc.measure(address_qubits, cbits)

    #qc = qc.reverse_bits()

    return qc


def run_circuit(boards, qc_generator, draw=False):
    qc = qc_generator(boards)

    if draw:
        qc.draw(output='mpl')
        plt.show()

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    count = result.get_counts()

    for k in sorted(count, key=count.get, reverse=True):
        print(f"Solution: {k} (Board {int(''.join(reversed(k)), 2)})\tCounts: {count[k]}")


def compute_circuit_cost(boards, qc_generator, draw=False):
    from utils.grader_utils import compute_cost

    qc = qc_generator(boards)
    print(compute_cost(qc))


if __name__ == "__main__":

    #run_circuit(board2_bitstrings(problem_set3x3, board_size=3), lambda l: downsized(l, num_iterations=1))

    compute_circuit_cost(board2_bitstrings(problem_set), lambda l: week3_ans_func(l, num_iterations=1))
    run_circuit(board2_bitstrings(problem_set), lambda l: week3_ans_func(l, num_iterations=1))
