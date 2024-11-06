# https://github.com/mgrzesiuk/qiskit-check/blob/74109e85011ded43230f66a62cdf0e9dabf0e1f4/tst/test_engine/concerete_property_test/test_concrete_property.py
from abc import ABC
from typing import Sequence, Union

from pytest_mock import MockFixture
from qiskit import QuantumCircuit

from qiskit_check.test_engine.concrete_property_test import ConcretePropertyTest
from qiskit_check.test_engine.concrete_property_test.concrete_property_test import ConcretePropertyTestIterator
from qiskit_check.property_test import PropertyTest
from qiskit_check.property_test.assertions import AbstractAssertion
from qiskit_check.property_test.resources import Qubit


class ExamplePropertyTest(PropertyTest, ABC):
    @property
    def circuit(self) -> QuantumCircuit:
        return None

    def get_qubits(self) -> Sequence[Qubit]:
        return []

    def assertions(self, qubits: Sequence[Qubit]) -> Union[AbstractAssertion, Sequence[AbstractAssertion]]:
        return []

    @staticmethod
    def confidence_level() -> float:
        return 0.99

    @staticmethod
    def num_measurements() -> int:
        return 5

    @staticmethod
    def num_experiments() -> int:
        return 5


class NegativeExamplePropertyTest(ExamplePropertyTest):
    @staticmethod
    def num_test_cases() -> int:
        return -1


class ZeroExamplePropertyTest(ExamplePropertyTest):
    @staticmethod
    def num_test_cases() -> int:
        return 0


class TestConcretePropertyTest:
    def test_iterator_doesnt_iterate_when_negative_test_count(self, mocker: MockFixture):
        property_test = ConcretePropertyTest(NegativeExamplePropertyTest, mocker.MagicMock(), mocker.MagicMock())
        count = 0
        for _ in property_test:
            count += 1
        assert count == 0

    def test_iterator_doesnt_iterate_when_0_test_count(self, mocker: MockFixture):
        property_test = ConcretePropertyTest(ZeroExamplePropertyTest, mocker.MagicMock(), mocker.MagicMock())
        count = 0
        for _ in property_test:
            count += 1
        assert count == 0

    def test_iter_returns_proper_iterator(self, mocker: MockFixture):
        property_test = ConcretePropertyTest(ZeroExamplePropertyTest, mocker.MagicMock(), mocker.MagicMock())
        assert isinstance(property_test.__iter__(), ConcretePropertyTestIterator)
