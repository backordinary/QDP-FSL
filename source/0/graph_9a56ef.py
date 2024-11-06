# https://github.com/svenpruefer/quantumcomputing/blob/e082b6b829ccabdf1c9c64b5cc310ba8feaad2d5/qsoc/graph/graph.py
# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.

from typing import Set, Dict, Tuple, List, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers import BaseBackend, BaseJob

from qsoc.circuits.coloring import VertexColor, add_4_coloring_grover, get_color_from_binary_string


class Graph:

    def __init__(self, vertices: List[str], edges: Set[Tuple[str, str]],
                 given_colors: Dict[str, VertexColor]):
        """
        Constructor for graphs for which we want to solve the four-color problem using a quantum computer.
        :param vertices: Set of vertices of the graph. Needs to be less than 12.
        :param edges: Set of edges of the graph. Needs to be less than 25.
        :param given_colors: Any given colors.
        """
        # We consider only simple graphs that are not too large and whose edges and given colors are meaningful.
        if len(vertices) > 11:
            raise ValueError(f"Can only consider graphs with less than 12 vertices, but got {len(vertices)}")
        if len(edges) > 24:
            raise ValueError(f"Can only consider graphs with less than 25 edges, but got {len(edges)}")
        for edge_endpoint in list(sum(edges, ())):
            if edge_endpoint not in vertices:
                raise ValueError(f"Endpoint {edge_endpoint} of an edge is not contained in the set of vertices.")
        for vertex in given_colors.keys():
            if vertex not in vertices:
                raise ValueError(f"Vertex {vertex} with a specified color is not contained in the set of vertices.")
        # Instance variables
        self._vertices = vertices
        self._external_vertices = set()
        self._internal_vertices = []
        self._edges = edges
        self._external_edges = set()
        self._internal_edges = set()
        self._colors = {}
        self._vertex_registers: Dict[str, QuantumRegister] = {}
        self._ancilla_register: Optional[QuantumRegister] = None

        # Separate vertices
        for vertex in vertices:
            if vertex in given_colors.keys():
                self._external_vertices.add(vertex)
            else:
                self._internal_vertices.append(vertex)
        # Separate edges
        for edge in edges:
            if edge[0] in given_colors.keys() and edge[1] in given_colors.keys():
                if given_colors[edge[0]] == given_colors[edge[1]]:
                    raise ValueError(
                        f"Invalid input as the given colors are inconsistent since {edge[0]} and {edge[1]}" +
                        f" have an identical color {given_colors[edge[0]]}")
            elif edge[0] in given_colors.keys():
                self._external_edges.add(edge)
                self._colors[edge[0]] = given_colors[edge[0]]
            elif edge[1] in given_colors.keys():
                self._external_edges.add(edge)
                self._colors[edge[1]] = given_colors[edge[1]]
            else:
                self._internal_edges.add(edge)

        print(f"Created graph with {len(self._internal_vertices)} uncolored vertices,"
              f" {len(self._internal_edges)} internal edges and {len(self._external_edges)} external edges")

    def get_4_color_grover_circuit(self, repetitions: int = 5) -> QuantumCircuit:
        """
        Create a 4-color Grover quantum circuit with 'repetitions' many repetitions for this graph.

        Vertex quantum registers are named as 'v-<name>', where <name> is the name of the vertex and
        the ancilla qubit is called 'ancilla'.

        :return: Quantum Circuit with 4-color Grover circuit
        """
        # Create quantum registers for vertices
        vertices: Dict[str, QuantumRegister] = {name: QuantumRegister(2, name=f"v_{name}") for name in
                                                self._internal_vertices}
        self._vertex_registers = vertices

        # Create quantum register for ancilla qubit
        ancilla: QuantumRegister = QuantumRegister(1, name="ancilla")
        self._ancilla_register = ancilla

        # Determine colors for external edges
        external_edges: Set[Tuple[str, VertexColor]] = set()
        for v1, v2 in self._external_edges:
            if v1 in self._colors.keys():
                external_edges.add((v2, self._colors[v1]))
            if v2 in self._colors.keys():
                external_edges.add((v1, self._colors[v2]))

        # Determine how many target qubits are needed and create suitable quantum register
        # We separate internal and external edges into groups of four and add one qubit for
        # possibly remaining internal and external edges
        internal_number_4_groups, internal_remainder = divmod(len(self._internal_edges), 4)
        external_number_4_groups, external_remainder = divmod(len(external_edges), 4)
        total_number_target_qubits = internal_number_4_groups + external_number_4_groups
        if internal_remainder > 0:
            total_number_target_qubits += 1
        if external_remainder > 0:
            total_number_target_qubits += 1
        target: QuantumRegister = QuantumRegister(total_number_target_qubits, name="target")

        # Determine how many auxiliary qubits are needed and create suitable quantum register
        # The number of auxiliary qubits needed is the maximum number of necessary ancilla qubits
        # for any of the used operations. The needs are as follows:
        # - internal 4 edges: 6
        # - internal 3 edges: 4
        # - internal 2 edges: 2
        # - internal 1 edge: 0
        # - external 4 edges: 6
        # - external 3 edges: 4
        # - external 2 edges: 2
        # - external 1 edge: 0
        # - Multi-Toffoli-gate for combining the target qubits via AND: #target - 2
        # - Grover reflection: 2 * #vertices - 3 - #target
        #   (Because after the Grover oracle step, the target qubits are |0> and can be used as ancilla
        #   qubits for the Grover reflection)
        total_number_auxiliary_qubits = max(6,
                                            total_number_target_qubits - 2,
                                            2 * len(vertices) - 3 - total_number_target_qubits
                                            )
        auxiliary: QuantumRegister = QuantumRegister(total_number_auxiliary_qubits, name="auxiliary")

        # Create QuantumCircuit including all quantum registers
        qc: QuantumCircuit = QuantumCircuit(name="four_color_grover_circuit")
        for register in vertices.values():
            qc.add_register(register)
        qc.add_register(auxiliary)
        qc.add_register(target)
        qc.add_register(ancilla)

        print(
            f"Created quantum circuit for graph with {len(vertices)} vertex registers,"
            f" {total_number_auxiliary_qubits} auxiliary qubits,"
            f" {total_number_target_qubits} target qubits and {len(list(ancilla))} ancilla qubits")

        # Add 4-color Grover circuit
        add_4_coloring_grover(qc, vertices, self._internal_edges, external_edges, auxiliary, target, ancilla[0],
                              repetitions)

        return qc

    def get_4_color_grover_algorithm_with_measurements(self, repetitions: int = 5) -> QuantumCircuit:
        qc = self.get_4_color_grover_circuit(repetitions)
        self.add_measurements(qc)
        return qc

    def run_4_color_grover_algorithm(self,
                                     backend: BaseBackend,
                                     runs: int,
                                     repetitions: int = 5) -> Dict[str, int]:
        """
        Run a 4-color Grover algorithm for the graph and obtain observed results.

        :param backend: Backend to run the circuit on.
        :param runs: Number of times the simulation gets run.
        :param repetitions: Number of repetitions of Grover oracle and Grover reflection
        :return: A set of solutions, where each solution specifies the color of all internal vertices
        """
        # Create the quantum circuit
        qc: QuantumCircuit = self.get_4_color_grover_circuit(repetitions)

        # Add classical registers and measure values
        self.add_measurements(qc)

        # Execute algorithm
        job: BaseJob = execute(qc, backend, shots=runs)
        job_result = job.result()

        return job_result.get_counts(qc)

    def add_measurements(self, qc: QuantumCircuit) -> None:
        measure_registers: Dict[str, ClassicalRegister] = {vertex: ClassicalRegister(2, name=f"v_{vertex}_measure") for
                                                           vertex in self._internal_vertices}
        for vertex in self._internal_vertices:
            qc.add_register(measure_registers[vertex])
            qc.measure(self._vertex_registers[vertex], measure_registers[vertex])

    def run_4_cover_grover_algorithm_and_interpret_results(self,
                                                           backend: BaseBackend,
                                                           runs: int,
                                                           repetitions: int = 5) -> List[Dict[str, VertexColor]]:
        """
        Run a 4-color Grover algorithm for the graph and obtain filtered and interpreted results.

        :param backend: Backend to run the circuit on.
        :param runs: Number of times the simulation gets run.
        :param repetitions: Number of repetitions of Grover oracle and Grover reflection
        :return: A set of solutions, where each solution specifies the color of all internal vertices
        """
        results: Dict[str, int] = self.run_4_color_grover_algorithm(backend, runs, repetitions)
        return self.translate_result_to_dict(self.filter_result(results))

    def filter_result(self, result: Dict[str, int]) -> Set[str]:
        total_runs: int = sum(result.values())
        number_internal_vertices: int = len(self._internal_vertices)
        filtered_result: Set[str] = {binary_result for binary_result, absolute_count in result.items() if
                                     absolute_count > total_runs / (4 ** number_internal_vertices)}
        return filtered_result

    def translate_result_to_dict(self, filtered_result: Set[str]) -> List[Dict[str, VertexColor]]:
        number_internal_vertices: int = len(self._internal_vertices)
        result: List[Dict[str, VertexColor]] = []
        for solution_string in filtered_result:
            solution: Dict[str, VertexColor] = {}
            for i, binary in enumerate(solution_string.split(' ')):
                solution[self._internal_vertices[number_internal_vertices - i - 1]] = get_color_from_binary_string(
                    binary)
            if solution not in result:
                result.append(solution)
        return result

    def translate_result_to_string(self, filtered_result: Set[str]) -> Set[str]:
        """
        Translate results to readable strings of integers.

        THIS NEEDS ALL VERTEX NAMES TO BE INTEGERS.

        :param filtered_result:
        :return:
        """
        results_as_dict: List[Dict[str, VertexColor]] = self.translate_result_to_dict(filtered_result)
        result: Set[str] = set()
        for result_dict in results_as_dict:
            all_colors: Dict[str, VertexColor] = {**self._colors, **result_dict}
            result_dict_with_integers: Dict[int, int] = {int(vertex): color.value for vertex, color in
                                                         all_colors.items()}
            temp_result = ""
            for k, v in sorted(result_dict_with_integers, key=lambda x: x[0]):
                temp_result + str(v)
            result.add(temp_result)
        return result
