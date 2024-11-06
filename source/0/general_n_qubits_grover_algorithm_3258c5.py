# https://github.com/ping-shuo-hao/n_qubits_grover_algorithm/blob/0c8dc7b5708ae9464f2900715ff18cae621c26e4/general_n_qubits_grover_algorithm.py
# Created by Shuohao Ping
# Revised by Yunlun Li

import numpy as np
import random
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMT
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

number_of_qubits =4
list_of_possible_states = ['{:0{}b}'.format(i, number_of_qubits) for i in range(2 ** number_of_qubits)]

def select_target(number_of_target=1, lst=list_of_possible_states, manual=[]):
    for i in manual:
        if i < 0 or i >= len(lst):
            return []

    if len(manual) > 0:
        return [lst[i] for i in set(manual)]

    if number_of_target <= 0 or number_of_target > len(lst):
        return []

    return random.sample(list_of_possible_states, number_of_target)


def oracle_for_single_target(target, circuit):
    list_of_bits = list(target)

    for i in range(len(target)):
        if target[len(target)-1-i]=="0":
            circuit.x(i)

    cnz=MCMT('z', num_ctrl_qubits=len(target)-1, num_target_qubits=1)
    circuit.append(cnz,[i for i in range(len(target))])

    for i in range(len(target)):
        if target[len(target)-1-i]=="0":
            circuit.x(i)

def orcale_n_target(lst_of_target,circuit):
    for target in lst_of_target:
        oracle_for_single_target(target,circuit)

def create_amiplifer(circuit,number_of_qubits):
    for i in range(number_of_qubits):
        circuit.h(i)

    for i in range(number_of_qubits):
        circuit.x(i)

    cnz=MCMT('z', num_ctrl_qubits=number_of_qubits-1, num_target_qubits=1)
    circuit.append(cnz,[i for i in range(number_of_qubits)])

    for i in range(number_of_qubits):
        circuit.x(i)

    for i in range(number_of_qubits):
        circuit.h(i)

def grover_circuit(number_of_qubits,lst_of_targets=None,number_of_targets=1,repetition=1):

    circuit = QuantumCircuit(number_of_qubits, number_of_qubits)

    for i in range(number_of_qubits):
        circuit.h(i)

    for i in range(repetition):
        orcale_n_target(targets,circuit)
        create_amiplifer(circuit, number_of_qubits)

    circuit.measure([i for i in range(number_of_qubits)], [i for i in range(number_of_qubits)])

    return circuit



# Use Aer's qasm_simulator
simulator = QasmSimulator()

targets = select_target(manual=[2])

circuit=grover_circuit(number_of_qubits,[0],repetition=2)
# Map the quantum measurement to the classical bit

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)
print("Marked States:",targets)
print(circuit)
print(counts)

