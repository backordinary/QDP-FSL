# https://github.com/bartek-bartlomiej/master-thesis/blob/69e97f35a259d84654b107d44b524a3340414389/implementations/mix.py
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import QFT

from gates.mix.modular_exponentiation import modular_exponentiation_gate, controlled_modular_multiplication_gate
from implementations.shor import Shor


class MixShor(Shor):
    def _construct_circuit_with_semiclassical_QFT(self, a: int, N: int, n: int) -> QuantumCircuit:
        self._qft = QFT(n, do_swaps=False).to_gate()
        self._iqft = self._qft.inverse()

        return super()._construct_circuit_with_semiclassical_QFT(a, N, n)

    def _get_aux_register_size(self, n: int) -> int:
        return n + 1

    @property
    def _prefix(self) -> str:
        return 'Mix'

    def _modular_exponentiation_gate(self, constant: int, N: int, n: int) -> Instruction:
        return modular_exponentiation_gate(constant, N, n)

    def _modular_multiplication_gate(self, constant: int, N: int, n: int) -> Instruction:
        return controlled_modular_multiplication_gate(constant, N, n, self._qft, self._iqft)