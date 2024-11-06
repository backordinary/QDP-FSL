# https://github.com/mgrzesiuk/qiskit-check/blob/f06df70750eb58b685825aa403c58b5675dcbe75/e2e_test/entangled_assertion_test/tst.py
from abc import ABC
from typing import Collection, Sequence

from qiskit import QuantumCircuit

from e2e_test.base_property_test import BasePropertyTest
from qiskit_check.property_test.assertions import AbstractAssertion, AssertEntangled
from qiskit_check.property_test.resources import Qubit, QubitRange


class AbstractEntanglePropertyPropertyTest(BasePropertyTest, ABC):
    def get_qubits(self) -> Collection[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0)) for _ in range(2)]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertEntangled(qubits[0], qubits[1])


class EntanglePropertyTest(AbstractEntanglePropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc
