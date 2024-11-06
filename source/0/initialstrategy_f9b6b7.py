# https://github.com/magn5452/QiskitQaoa/blob/859d0d67e31f86191a28f99d00bb10197d846d05/VehicleRouting/standard/qaoa/InitialStrategy.py
from operator import xor

import numpy as np
from qiskit import QuantumCircuit
from qiskit.extensions import HamiltonianGate
from qiskit_nature.operators.second_quantization import SpinOp

from VehicleRouting.framework.qaoa.CircuitStrategy import InitialStrategy, MixerStrategy
from VehicleRouting.standard.qaoa.MixerStrategy import PartialSwapMixerStrategy, AdjacentSwapMixerStrategy


class NullInitialStrategy(InitialStrategy):

    def __init__(self):
        pass

    def set_up_initial_state_circuit(self, quantum_circuit: QuantumCircuit):
        pass


class HGateInitialStrategy(InitialStrategy):

    def __init__(self):
        pass

    def set_up_initial_state_circuit(self, quantum_circuit: QuantumCircuit):
        quantum_circuit.barrier()
        for index_qubit in range(0, quantum_circuit.num_qubits):
            quantum_circuit.h(index_qubit)


class OneHotSingleInitialStrategy(InitialStrategy):

    def __init__(self):
        pass

    def set_up_initial_state_circuit(self, quantum_circuit: QuantumCircuit):
        quantum_circuit.barrier()
        initial_array = self.get_initial_array(quantum_circuit.num_qubits)

        for index_qubit, value_qubit in enumerate(initial_array):
            if value_qubit == 1:
                quantum_circuit.x(index_qubit)

    def get_initial_array(self, num_qubits):
        initial_array = np.zeros(num_qubits)
        num_vertices = int(np.round(np.sqrt(num_qubits)))
        for index_qubit in range(0, num_qubits):
            if np.mod(index_qubit, num_vertices + 1) == 0:
                initial_array[index_qubit] = 1
        return initial_array


class XGateInitialStrategy(InitialStrategy):

    def __init__(self, initial_array):
        self.initial_array = initial_array

    def set_up_initial_state_circuit(self, quantum_circuit: QuantumCircuit):
        quantum_circuit.barrier()
        for index_qubit, value_qubit in enumerate(self.initial_array):
            if value_qubit == 1:
                quantum_circuit.x(index_qubit)


class PartialSwapInitialStrategy(InitialStrategy):

    def __init__(self):
        pass

    def set_up_initial_state_circuit(self, quantum_circuit):
        XGateInitialStrategy(initial_array=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])).set_up_initial_state_circuit(
            quantum_circuit)

        quantum_circuit.barrier()

        n = 0
        beta_3 = n * np.pi - np.arcsin(np.sqrt(1 / 3))
        beta_2 = n * np.pi - np.arcsin(np.sqrt(1 / 2))

        self.set_up_partial_swap_gate(beta_2, 1, 2, 1, 2, quantum_circuit)
        self.set_up_partial_swap_gate(beta_3, 0, 1, 0, 1, quantum_circuit)
        self.set_up_partial_swap_gate(beta_3, 0, 2, 0, 1, quantum_circuit)
        self.set_up_partial_swap_gate(beta_2, 0, 2, 0, 2, quantum_circuit)
        self.set_up_partial_swap_gate(beta_2, 0, 1, 0, 2, quantum_circuit)

    def set_up_partial_swap_gate(self, beta, city_u, city_v, position_i, position_j, quantum_circuit):
        num_vertices = int(np.round(np.sqrt(quantum_circuit.num_qubits)))
        hamiltonian = SpinOp([("++--", 1), ("--++", 1)])
        unitary_gate = HamiltonianGate(hamiltonian.to_matrix(), beta, label="(" + str(round(beta, 1)) + ")")
        quantum_circuit.append(unitary_gate, [self.map(city_u, position_i, num_vertices),
                                              self.map(city_v, position_j,
                                                       num_vertices),
                                              self.map(city_u, position_j,
                                                       num_vertices),
                                              self.map(city_v, position_i, num_vertices)])

    def map(self, city: int, position: int, num_vertices: int):
        mapping = num_vertices * np.mod(city, num_vertices) + np.mod(position, num_vertices)
        return mapping

class CustomInitialStrategy(InitialStrategy):

    def __init__(self):
        pass

    def set_up_initial_state_circuit(self, quantum_circuit: QuantumCircuit):
        initializer = OneHotSingleInitialStrategy()
        initializer.set_up_initial_state_circuit(quantum_circuit)
        mixer = AdjacentSwapMixerStrategy()
        mixer.set_up_mixer_circuit(np.pi / 2, quantum_circuit)


class AdjacentSwapInitialStrategy(InitialStrategy):

    def __init__(self):
        pass

    def set_up_initial_state_circuit(self, quantum_circuit):
        initializer = OneHotSingleInitialStrategy()
        initializer.set_up_initial_state_circuit(quantum_circuit)
        initial_array = initializer.get_initial_array(quantum_circuit.num_qubits)

        quantum_circuit.barrier()
        num_vertices = int(np.round(np.sqrt(quantum_circuit.num_qubits)))
        hamiltonian = self.make_adjacent_swap_mixer_hamiltonian(num_vertices, initial_array)
        beta = np.sqrt(3) / (3 ** 2) * np.pi  # for n = 3 sqrt(3)/3^2
        unitary_gate = HamiltonianGate(hamiltonian.to_matrix(), beta, label="(" + str(round(beta, 1)) + ")")
        quantum_circuit.append(unitary_gate, range(quantum_circuit.num_qubits))

    def make_adjacent_swap_mixer_hamiltonian(self, num_vertices, initial_array) -> SpinOp:

        opr_list = []
        for city_u in range(num_vertices):
            for city_v in range(num_vertices):
                for position_i in range(num_vertices):
                    for position_j in range(num_vertices):
                        if city_u < city_v and position_i < position_j:
                            if initial_array[self.map(city_u, position_i, num_vertices)] == 1 and initial_array[
                                self.map(city_v, position_j, num_vertices)] == 1:
                                string, string_hermitian = self.make_string_partial_mixer(city_u, city_v, position_i,
                                                                                          position_j, num_vertices)
                                print(string)
                                opr_list.append((string, 1))
                                opr_list.append((string_hermitian, 1))

        return SpinOp(opr_list)

    def make_string_partial_mixer(self, city_u, city_v, position_i, position_j, num_vertices):

        string_list = []
        string_hermitian_list = []
        for i in range(num_vertices ** 2):
            string_list.append("I")
            string_hermitian_list.append("I")

        string_list[self.map(city_u, position_i, num_vertices)] = "+"
        string_list[self.map(city_v, position_j, num_vertices)] = "+"
        string_list[self.map(city_u, position_j, num_vertices)] = "-"
        string_list[self.map(city_v, position_i, num_vertices)] = "-"

        string_hermitian_list[self.map(city_u, position_i, num_vertices)] = "-"
        string_hermitian_list[self.map(city_v, position_j, num_vertices)] = "-"
        string_hermitian_list[self.map(city_u, position_j, num_vertices)] = "+"
        string_hermitian_list[self.map(city_v, position_i, num_vertices)] = "+"

        new_string = "".join(string_list)
        new_string_hermitian = "".join(string_hermitian_list)
        return new_string, new_string_hermitian

    def map(self, city: int, position: int, num_vertices: int):
        mapping = num_vertices * np.mod(city, num_vertices) + np.mod(position, num_vertices)
        return mapping


class Adjacent2SwapInitialStrategy(InitialStrategy):

    def __init__(self):
        pass

    def set_up_initial_state_circuit(self, quantum_circuit):
        initializer = OneHotSingleInitialStrategy()
        initializer.set_up_initial_state_circuit(quantum_circuit)
        initial_array = initializer.get_initial_array(quantum_circuit.num_qubits)

        quantum_circuit.barrier()
        num_vertices = int(np.round(np.sqrt(quantum_circuit.num_qubits)))
        hamiltonian = self.make_adjacent_swap_mixer_hamiltonian(num_vertices, initial_array)
        beta = np.sqrt(3) / (3 ** 2) * np.pi  # for n = 3 sqrt(3)/3^2
        unitary_gate = HamiltonianGate(hamiltonian.to_matrix(), beta, label="(" + str(round(beta, 1)) + ")")
        quantum_circuit.append(unitary_gate, range(quantum_circuit.num_qubits))

    def make_adjacent_swap_mixer_hamiltonian(self, num_vertices, initial_array) -> SpinOp:

        opr_list = []
        for city_u in range(num_vertices):
            for city_v in range(num_vertices):
                for position_i in range(num_vertices):
                    for position_j in range(num_vertices):
                        if city_u != city_v and position_i != position_j:
                            if initial_array[self.map(city_u, position_i, num_vertices)] == 1 and initial_array[
                                self.map(city_v, position_j, num_vertices)] == 1:
                                string, string_hermitian = self.make_string_partial_mixer(city_u, city_v, position_i,
                                                                                          position_j, num_vertices)
                                print(string)
                                opr_list.append((string, 1))
                                opr_list.append((string_hermitian, 1))

        return SpinOp(opr_list)

    def make_string_partial_mixer(self, city_u, city_v, position_i, position_j, num_vertices):

        string_list = []
        string_hermitian_list = []
        for i in range(num_vertices ** 2):
            string_list.append("I")
            string_hermitian_list.append("I")

        string_list[self.map(city_u, position_i, num_vertices)] = "+"
        string_list[self.map(city_v, position_j, num_vertices)] = "+"
        string_list[self.map(city_u, position_j, num_vertices)] = "-"
        string_list[self.map(city_v, position_i, num_vertices)] = "-"

        string_hermitian_list[self.map(city_u, position_i, num_vertices)] = "-"
        string_hermitian_list[self.map(city_v, position_j, num_vertices)] = "-"
        string_hermitian_list[self.map(city_u, position_j, num_vertices)] = "+"
        string_hermitian_list[self.map(city_v, position_i, num_vertices)] = "+"

        new_string = "".join(string_list)
        new_string_hermitian = "".join(string_hermitian_list)
        return new_string, new_string_hermitian

    def map(self, city: int, position: int, num_vertices: int):
        mapping = num_vertices * np.mod(city, num_vertices) + np.mod(position, num_vertices)
        return mapping
