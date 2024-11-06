# https://github.com/mgrzesiuk/qiskit-check/blob/6c81ce8075291e51434f7ba71ef15710343d9874/e2e_test/equal_assertion_test/test_equal.py
from abc import ABC
from typing import Sequence

from qiskit import QuantumCircuit

from e2e_test.base_property_test import BasePropertyTest
from qiskit_check.property_test.assertions import AbstractAssertion
from qiskit_check.property_test.assertions import AssertEqualByProbability
from qiskit_check.property_test.resources.test_resource import Qubit
from qiskit_check.property_test.resources.qubit_range import QubitRange


class AbstractEqualPropertyPropertyTest(BasePropertyTest, ABC):
    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0)), Qubit(QubitRange(0, 0, 0, 0))]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertEqualByProbability(qubits[0], qubits[1])


class XEqualPropertyTest(AbstractEqualPropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.measure_all()
        return qc


class HEqualPropertyTest(AbstractEqualPropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.measure_all()
        return qc


class SEqualPropertyTest(AbstractEqualPropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.s(0)
        qc.s(1)
        qc.measure_all()
        return qc


class DoubleHEqualPropertyTest(AbstractEqualPropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.x(1)
        qc.x(1)
        qc.measure_all()
        return qc
