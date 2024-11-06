# https://github.com/mgrzesiuk/qiskit-check/blob/f06df70750eb58b685825aa403c58b5675dcbe75/case_studies/phase_estimation/tst.py
from abc import ABC
from math import pi
from typing import Collection, Sequence, List
from random import choice

from qiskit import QuantumCircuit

from case_studies.example_test_base import ExampleTestBase
from case_studies.phase_estimation.src import phase_estimation
from qiskit_check.property_test.assertions import AbstractAssertion, AssertProbability
from qiskit_check.property_test.resources.test_resource import Qubit
from qiskit_check.property_test.resources.qubit_range import QubitRange


class AbstractPhaseEstimationPropertyTest(ExampleTestBase, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.phase = choice([pi/4, pi/2, pi])

    def get_qubits(self) -> Collection[Qubit]:
        return [
            Qubit(QubitRange(0, 0, 0, 0)),
            Qubit(QubitRange(0, 0, 0, 0)),
            Qubit(QubitRange(0, 0, 0, 0)),
            Qubit(QubitRange(0, 0, 0, 0))
        ]

    def assertions(self, qubits: Sequence[Qubit]) -> List[AbstractAssertion]:
        if self.phase == pi / 4:
            return [
                AssertProbability(self.qubits[0], "1", 1),
                AssertProbability(self.qubits[1], "0", 1),
                AssertProbability(self.qubits[2], "0", 1),
            ]
        elif self.phase == pi / 2:
            return [
                AssertProbability(self.qubits[0], "0", 1),
                AssertProbability(self.qubits[1], "1", 1),
                AssertProbability(self.qubits[2], "0", 1),
            ]
        if self.phase == pi:
            return [
                AssertProbability(self.qubits[0], "0", 1),
                AssertProbability(self.qubits[1], "0", 1),
                AssertProbability(self.qubits[2], "1", 1),
            ]


class PhaseEstimationPropertyTest(AbstractPhaseEstimationPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        return phase_estimation(QuantumCircuit(len(self.qubits), 3), self.phase)
