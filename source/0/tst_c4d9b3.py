# https://github.com/mgrzesiuk/qiskit-check/blob/6c81ce8075291e51434f7ba71ef15710343d9874/case_studies/teleportation/tst.py
from abc import ABC
from typing import Sequence

from qiskit import QuantumCircuit

from case_studies.example_test_base import ExampleTestBase
from case_studies.teleportation.src import quantum_teleportation
from qiskit_check.property_test.assertions import AbstractAssertion, AssertTeleportedByProbability
from qiskit_check.property_test.resources.test_resource import Qubit
from qiskit_check.property_test.resources.qubit_range import QubitRange, AnyRange


class AbstractTeleportationProperty(ExampleTestBase, ABC):
    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(AnyRange()), Qubit(QubitRange(0, 0, 0, 0)), Qubit(QubitRange(0, 0, 0, 0))]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertTeleportedByProbability(qubits[0], qubits[2])

    @staticmethod
    def num_test_cases() -> int:
        return 10


class TeleportationProperty(AbstractTeleportationProperty):
    @property
    def circuit(self) -> QuantumCircuit:
        return quantum_teleportation()
