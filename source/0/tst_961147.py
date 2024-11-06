# https://github.com/mgrzesiuk/qiskit-check/blob/f06df70750eb58b685825aa403c58b5675dcbe75/case_studies/superdense_coding/tst.py
from abc import ABC
from random import choices
from typing import Sequence, List

from qiskit import QuantumCircuit

from case_studies.example_test_base import ExampleTestBase
from case_studies.superdense_coding.src import superdense_coding
from qiskit_check.property_test.assertions import AbstractAssertion, AssertProbability
from qiskit_check.property_test.resources.test_resource import Qubit
from qiskit_check.property_test.resources.qubit_range import QubitRange


class AbstractSuperdenseCodingPropertyTest(ExampleTestBase, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.bitstring = choices(['11', '10', '01', '00'])[0]

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0)), Qubit(QubitRange(0, 0, 0, 0))]

    def assertions(self, qubits: Sequence[Qubit]) -> List[AbstractAssertion]:
        return [
            AssertProbability(self.qubits[0], self.bitstring[1], 1),
            AssertProbability(self.qubits[1], self.bitstring[0], 1)
        ]


class SuperdenseCodingPropertyTest(AbstractSuperdenseCodingPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        return superdense_coding(self.bitstring)
