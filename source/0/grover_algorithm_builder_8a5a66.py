# https://github.com/QuentinPrigent/quantum-computing-project/blob/94db92caef29cd8529b1c9fcc8e6c488913c1f55/src/builders/grover_algorithm_builder.py
import os
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracle, GroverOperator


class GroverAlgorithmBuilder:
    def __init__(self):
        self.number_of_qubits = None

        self.oracle_quantum_circuit = None
        self.grover_algorithm_quantum_circuit = None

    def oracle_quantum_circuit_builder(self):
        project_directory = os.getcwd()
        self.oracle_quantum_circuit = PhaseOracle.from_dimacs_file(project_directory + '/files/search.dimacs')
        self.number_of_qubits = self.oracle_quantum_circuit.num_qubits

    def grover_algorithm_quantum_circuit_builder(self):
        self.grover_algorithm_quantum_circuit = QuantumCircuit(self.number_of_qubits)
        self.grover_algorithm_quantum_circuit.h([index for index in range(self.number_of_qubits)])
        grover_operator = GroverOperator(self.oracle_quantum_circuit)
        self.grover_algorithm_quantum_circuit = self.grover_algorithm_quantum_circuit.compose(grover_operator)
        self.grover_algorithm_quantum_circuit.measure_all()
