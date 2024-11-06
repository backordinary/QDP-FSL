# https://github.com/mgrzesiuk/qiskit-check/blob/6c81ce8075291e51434f7ba71ef15710343d9874/e2e_test/state_transformed_assertion_test/test_state_transformed.py
from math import pi
from typing import Sequence, List

from qiskit import QuantumCircuit
from scipy.spatial.transform import Rotation

from e2e_test.base_property_test import BasePropertyTest
from qiskit_check.property_test.assertions import AbstractAssertion
from qiskit_check.property_test.assertions import AssertTransformedByState
from qiskit_check.property_test.resources.test_resource import Qubit
from qiskit_check.property_test.resources.qubit_range import AnyRange


class SingleGateStateTransformedPropertyPropertyTest(BasePropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.h(0)
        return qc

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(AnyRange())]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertTransformedByState(qubits[0], Rotation.from_euler("X", [pi]), 1)


class SSingleGateStateTransformedPropertyPropertyTest(BasePropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.s(0)
        return qc

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(AnyRange())]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertTransformedByState(qubits[0], Rotation.from_euler("Z", [pi/2]), 1)

class HSingleGateStateTransformedPropertyPropertyTest(BasePropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        return qc

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(AnyRange())]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertTransformedByState(qubits[0], Rotation.from_euler("XY", [pi, pi/2]), 1)


class MultipleGateStateTransformedPropertyPropertyTest(BasePropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        qc.x(0)
        qc.h(0)
        return qc

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(AnyRange())]

    def assertions(self, qubits: Sequence[Qubit]) -> List[AbstractAssertion]:
        return [
            AssertTransformedByState(qubits[0], Rotation.from_euler("XY", [pi, pi/2]), 1),
            AssertTransformedByState(qubits[0], Rotation.identity(), 2),
            AssertTransformedByState(qubits[0], Rotation.from_euler("X", [pi]), 3),
            AssertTransformedByState(qubits[0], Rotation.from_euler("XYX", [pi, pi/2, pi]), 4),
            AssertTransformedByState(qubits[0], Rotation.from_euler("XYX", [pi, pi/2, pi]))
        ]
