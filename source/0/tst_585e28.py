# https://github.com/mgrzesiuk/qiskit-check/blob/6c81ce8075291e51434f7ba71ef15710343d9874/case_studies/deutsch_jozsa/tst.py
from abc import ABC
from random import randint
from typing import Collection, List, Sequence, Dict

from qiskit import QuantumCircuit

from case_studies.deutsch_jozsa.src import deutsch_jozsa
from case_studies.example_test_base import ExampleTestBase
from qiskit_check.property_test.assertions import AbstractAssertion, AssertTrue
from qiskit_check.property_test.resources import Qubit, QubitRange, ConcreteQubit


class AbstractDeutschJozsaPropertyTest(ExampleTestBase, ABC):
    def __init__(self):
        self.balanced_or_constant = randint(0, 1)
        super().__init__()

    def get_qubits(self) -> Collection[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0)) for _ in range(4)]

    def evaluate_correctness(self, measurement: List[Dict[str, int]], resource_matcher: Dict[Qubit, ConcreteQubit]) -> float:
        counts = measurement[0]
        measurement = list(counts.keys())[0][:-1]
        if len(counts.keys()) > 1:
            return 0
        if self.balanced_or_constant == 1:
            return measurement == "1"*(len(self.qubits)-1)
        else:
            return measurement == "0"*(len(self.qubits)-1)

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertTrue(qubits[:-1], self.evaluate_correctness, 1)


class DeutschJozsaPropertyTest(AbstractDeutschJozsaPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        return deutsch_jozsa(QuantumCircuit(4, 3), self.balanced_or_constant)
