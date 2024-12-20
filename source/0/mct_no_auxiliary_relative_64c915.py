# https://github.com/QuCoNot/QuCoNot/blob/0bd917ad435df1ea586be5b397bc2303cd00d82a/quconot/implementations/mct_no_auxiliary_relative.py
# Quconot/quconot/implementations/mct_no_auxiliary_relative_phase.py
#
# Authors:
#  - Handy Kurniawan
#
# Apply the implementation from Qiskit MCT

from copy import deepcopy
from typing import List

from qiskit import QuantumCircuit, transpile

from .mct_base import MCTBase


class MCTNoAuxiliaryRelative(MCTBase):
    def __init__(self, controls_no: int, **kwargs) -> None:
        if controls_no not in [2, 3]:
            raise ValueError("Number of controls must be 2 or 3 for this implementation")
        self._n = controls_no
        self._circuit: QuantumCircuit = None

    @classmethod
    def verify_mct_cases(
        self,
        controls_no: int,
        max_auxiliary: int,
        relative_phase: bool = False,
        clean_acilla: bool = True,
        wasted_auxiliary: bool = False,
        separable_wasted_auxiliary: bool = False,
    ) -> List["MCTBase"]:
        """Generate all possible MCT implementation satisfying the requirements

        relative_phase: true / false (D)
        clean_auxiliary: true (D) / false
        wasted_auxiliary: true / false (D)
        separable_wasted_auxiliary: true / false (D), requires wasted_auxiliary set to True

        :return: a quantum circuit
        :rtype: QuantumCircuit
        """
        return [MCTNoAuxiliaryRelative(controls_no)]

    def generate_circuit(self) -> QuantumCircuit:
        """Return a QuantumCircuit implementation

        :return: a quantum circuit
        :rtype: QuantumCircuit
        """
        qc = QuantumCircuit(self._n + 1)
        if self._n == 2:
            qc.rccx(0, 1, 2)
        elif self._n == 3:
            qc.rcccx(0, 1, 2, 3)

        # should be done for all implementations
        # TODO: solve issue with reordered qubits
        self._circuit = transpile(qc, basis_gates=["cx", "u3"])
        return deepcopy(self._circuit)

    def num_auxiliary_qubits(self):
        """Return number of auxiliary qubits

        :return: number of auxiliary qubits
        :rtype: int
        """
        return 0
