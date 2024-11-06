# https://github.com/pdebus/IBMQuantumChallenge2020/blob/366552f9f8129f73d5483f8c9d7f9df1afb8d929/src/board_oracles.py
from qiskit import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute

import numpy as np
import matplotlib.pyplot as plt

from utils.board_tools import compute_switch_edges
from board_qram import init_light_states
from circuit_parts.diffusers import diffuser_gate, diffuser


def lights_out_oracle(qc, switch_qubits, light_qubits, output_qubit, ancilla_qubits=None):

    num_lights = len(switch_qubits)

    if ancilla_qubits is not None:
        assert len(ancilla_qubits) >= len(switch_qubits) - 2

    edges = compute_switch_edges(int(np.sqrt(num_lights)))

    # Connect Switches to lights
    for source, targets in edges.items():
        for target in targets:
            qc.cx(switch_qubits[source], light_qubits[target])

    if ancilla_qubits is not None:
        qc.mct(light_qubits, output_qubit, ancilla_qubits, mode='basic')
    else:
        qc.mct(light_qubits, output_qubit)

    # Uncompute
    for source, targets in edges.items():
        for target in targets:
            qc.cx(switch_qubits[source], light_qubits[target])


def lights_out_oracle_gate(use_ancillas=False, num_lights=9):

    switch_qubits = QuantumRegister(num_lights, name='switch')
    light_qubits = QuantumRegister(num_lights, name='light')
    output_qubit = QuantumRegister(1, name='out')

    if use_ancillas:
        ancilla_qubits = QuantumRegister(num_lights - 2, name='ancilla')
        qc = QuantumCircuit(switch_qubits, light_qubits, output_qubit, ancilla_qubits)
    else:
        ancilla_qubits = None
        qc = QuantumCircuit(switch_qubits, light_qubits, output_qubit)

    lights_out_oracle(qc, switch_qubits, light_qubits, output_qubit, ancilla_qubits)

    oracle = qc.to_gate()
    oracle.name = "$O_{2a}$"

    return oracle


def single_board_solution(lights, num_iterations=1):
    print("\n===Solution with Instructions===")

    num_lights = len(lights)

    switch_qubits = QuantumRegister(num_lights, name='switch')
    light_qubits = QuantumRegister(num_lights, name='light')
    ancilla_qubits = QuantumRegister(num_lights - 2, name='ancilla')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(num_lights, name="cbits")

    qc = QuantumCircuit(switch_qubits, light_qubits, ancilla_qubits, output_qubit, cbits)

    # Init
    qc.x(output_qubit)
    qc.h(output_qubit)

    qc.h(switch_qubits)

    init_light_states(qc, lights, light_qubits)
    qc.barrier()

    for i in range(num_iterations):
        lights_out_oracle(qc, switch_qubits, light_qubits, output_qubit, ancilla_qubits)
        diffuser(qc, switch_qubits, ancilla_qubits)

    qc.measure(switch_qubits, cbits)
    qc = qc.reverse_bits()

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    count = result.get_counts()

    max_k = min(len(count), 10)

    for k in sorted(count, key=count.get, reverse=True)[:max_k]:
        print(f"Solution: {k}\tCounts: {count[k]}")

    qc.draw(output='mpl')
    plt.show()


def single_board_solution_gates(lights, num_iterations=1):
    print("\n===Solution with Gates===")

    num_lights = len(lights)

    switch_qubits = QuantumRegister(num_lights, name='switch')
    light_qubits = QuantumRegister(num_lights, name='light')
    ancilla_qubits = QuantumRegister(num_lights - 2, name='ancilla')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(num_lights, name="cbits")

    qc = QuantumCircuit(switch_qubits, light_qubits, ancilla_qubits, output_qubit, cbits)
    oracle = lights_out_oracle_gate(use_ancillas=True, num_lights=num_lights)
    diff = diffuser_gate(num_lights, use_ancillas=True)

    # Init
    qc.x(output_qubit)
    qc.h(output_qubit)

    qc.h(switch_qubits)

    init_light_states(qc, lights, light_qubits)
    qc.barrier()

    for i in range(num_iterations):
        qc.append(oracle, switch_qubits[:] + light_qubits[:] + output_qubit[:] + ancilla_qubits[:])
        qc.append(diff, switch_qubits[:] + ancilla_qubits[:])

    qc.measure(switch_qubits, cbits)
    qc = qc.reverse_bits()

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    count = result.get_counts()

    max_k = min(len(count), 10)

    for k in sorted(count, key=count.get, reverse=True)[:max_k]:
        print(f"Solution: {k}\tCounts: {count[k]}")

    qc.draw(output='mpl')
    plt.show()


if __name__ == "__main__":

    lights = [1, 0, 1, 1]
    print("\nGround thruth: 0010")
    single_board_solution(lights, num_iterations=1)
    single_board_solution_gates(lights, num_iterations=1)

    lights2 = [0, 0, 0, 1, 0, 1, 1, 1, 0]
    print("Ground thruth: 100110000")
    single_board_solution(lights2, num_iterations=1)
    single_board_solution_gates(lights2, num_iterations=1)

