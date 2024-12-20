# https://github.com/QuCoNot/QuCoNot/blob/0bd917ad435df1ea586be5b397bc2303cd00d82a/quconot/implementations/mct_vchain_dirty.py
# Quconot/quconot/implementations/mct_vchain_dirty.py
#
# Authors:
#  - Handy Kurniawan
#
# Apply the implementation from Qiskit MCT

from copy import deepcopy
from typing import List

from qiskit import QuantumCircuit, transpile

from .mct_base import MCTBase


class MCTVChainDirty(MCTBase):
    def __init__(self, controls_no: int, **kwargs) -> None:
        if controls_no < 2:
            raise ValueError("Number of controls must be >= 2 for this implementation")
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
        if max_auxiliary < controls_no - 2:
            return []  # if max_auxiliary allowed is to small - no representation given
        else:
            return [MCTVChainDirty(controls_no)]  # only one available

    def generate_circuit(self) -> QuantumCircuit:
        """Return a QuantumCircuit implementation

        :return: a quantum circuit
        :rtype: QuantumCircuit
        """
        qc = QuantumCircuit(2 * self._n - 1)
        qc.mct(
            list(range(self._n)),
            self._n,
            ancilla_qubits=list(range(self._n + 1, 2 * self._n - 1)),
            mode="v-chain-dirty",
        )

        # should be done for all implementations
        # TODO: solve issue with reordered qubits
        self._circuit = transpile(qc, basis_gates=["cx", "u3"])
        return deepcopy(self._circuit)

    def num_auxiliary_qubits(self):
        """Return number of auxiliary qubits

        :return: number of auxiliary qubits
        :rtype: int
        """
        return self._n - 2
