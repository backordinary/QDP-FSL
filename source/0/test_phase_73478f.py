# https://github.com/mgrzesiuk/qiskit-check/blob/74109e85011ded43230f66a62cdf0e9dabf0e1f4/e2e_test/phase_assertion_test/test_phase.py
from math import pi
from typing import Sequence

from qiskit import QuantumCircuit

from e2e_test.base_property_test import BasePropertyTest
from qiskit_check.property_test.assertions import AbstractAssertion
from qiskit_check.property_test.assertions.assert_phase import AssertPhase
from qiskit_check.property_test.resources import Qubit, QubitRange

class AssertPhaseNoPhasePropertyTest(BasePropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        return qc

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0))]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertPhase(qubits[0], 0)

class AssertPhasePropertyTest(BasePropertyTest):
    phase = pi/4
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(self.phase, 0)
        return qc

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0))]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertPhase(qubits[0], self.phase)

class AssertPhaseMidCircuitPropertyTest(BasePropertyTest):
    phase = pi/4
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(self.phase, 0)
        qc.p(self.phase, 0)
        return qc

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0))]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertPhase(qubits[0], self.phase, 2)
