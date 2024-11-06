# https://github.com/mgrzesiuk/qiskit-check/blob/f06df70750eb58b685825aa403c58b5675dcbe75/case_studies/grover_search/src.py
from math import ceil, pi, sqrt

from qiskit import QuantumCircuit
from qiskit.algorithms import AmplificationProblem, Grover


def oracle(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit.mcrz(pi, circuit.qubits[:-1], circuit.qubits[-1])
    return circuit


def grover_search(number_of_solutions: int, oracle_circuit: QuantumCircuit) -> QuantumCircuit:
    search_space_size = 2**len(oracle_circuit.qubits)

    number_of_rotations = ceil(pi*sqrt(number_of_solutions/search_space_size)/4)

    problem = AmplificationProblem(oracle_circuit, is_good_state=[])
    grover = Grover(iterations=number_of_rotations)
    circuit = grover.construct_circuit(problem)
    circuit.measure_all()
    return circuit


def mutation_oracle(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit.mcrz(pi, circuit.qubits[:-3], circuit.qubits[-1])
    return circuit


def mutation_wrong_gate_oracle(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit.mcrx(pi, circuit.qubits[:-1], circuit.qubits[-1])
    return circuit


def mutation_add_h_grover_search(number_of_solutions: int, oracle_circuit: QuantumCircuit) -> QuantumCircuit:
    search_space_size = 2**len(oracle_circuit.qubits)

    number_of_rotations = ceil(pi*sqrt(number_of_solutions/search_space_size)/4)

    problem = AmplificationProblem(oracle_circuit, is_good_state=[])
    grover = Grover(iterations=number_of_rotations)
    circuit = grover.construct_circuit(problem)
    for qubit in circuit.qubits:
        circuit.h(qubit)
    circuit.measure_all()
    return circuit
