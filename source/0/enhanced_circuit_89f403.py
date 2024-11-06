# https://github.com/mgrzesiuk/qiskit_utils/blob/a3c9b07de0f2d8acb05ba247ca792283f87c74d2/qiskit_utils/enhanced_circuit.py
from collections.abc import Sequence
from typing import Union

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Qubit, Clbit

from qiskit_utils.insert import insert_instruction


class QuantumCircuitEnhanced(QuantumCircuit):
    def insert(
            self, instruction: Instruction, qubits: Union[Sequence[Qubit, int]],
            clbits: Union[Sequence[Clbit, int]], index: int, in_place: bool = True) -> QuantumCircuit:
        """
        insert instruction at a specified place (in lists of instructions from circuit.data)
        :param instruction: instruction to be inserted
        :param qubits: qubits used for the instruction (can be indexes or objects
        :param clbits: clbits used for the instruction (can be indexes or objects)
        :param index: index where the instruction will be placed in self.data
        :param in_place: creates new circuit if False and returns it, otherwise updates self and returns it
        :return: circuit with instruction inserted
        """
        return insert_instruction(self, instruction, qubits, clbits, index, in_place=in_place)
