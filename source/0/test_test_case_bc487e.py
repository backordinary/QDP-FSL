# https://github.com/mgrzesiuk/qiskit-check/blob/74109e85011ded43230f66a62cdf0e9dabf0e1f4/tst/test_engine/concerete_property_test/test_test_case.py
from typing import Sequence, Union

import pytest
from pytest_mock import MockFixture
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_check.test_engine.assessor import AssessorFactory
from qiskit_check.test_engine.concrete_property_test.test_case import TestCaseGenerator
from qiskit_check.test_engine.generator import QubitInputGenerator
from qiskit_check.property_test import PropertyTest
from qiskit_check.property_test.assertions import AbstractAssertion
from qiskit_check.property_test.property_test_errors import IncorrectPropertyTestError
from qiskit_check.property_test.resources import Qubit, AnyRange


class PropertyTestExample(PropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        return QuantumCircuit(1)

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(AnyRange())]

    def assertions(self, qubits: Sequence[Qubit]) -> Union[AbstractAssertion, Sequence[AbstractAssertion]]:
        return []

    @staticmethod
    def confidence_level() -> float:
        return 0.99

    @staticmethod
    def num_test_cases() -> int:
        return 1

    @staticmethod
    def num_measurements() -> int:
        return 2

    @staticmethod
    def num_experiments() -> int:
        return 3


class WrongExamplePropertyTest(PropertyTestExample):
    def __init__(self, additional_argument):
        super().__init__()
        self.additional_argument = additional_argument


class TestTestCase:
    def test_generate_incorrect_property_test_error_raised_when_too_many_arguments_in_constructor(self, mocker: MockFixture):
        test_case_generator = TestCaseGenerator(WrongExamplePropertyTest, mocker.MagicMock(), mocker.MagicMock())
        with pytest.raises(IncorrectPropertyTestError):
            test_case_generator.generate()

    def test_generate_returns_correctly_initialized_test_case_when_correct_input(self, mocker: MockFixture):
        assessor_factor = mocker.patch.object(AssessorFactory, "build")
        assessor_factor.build.return_value = mocker.MagicMock()
        qubit_input_generator = mocker.patch.object(QubitInputGenerator, "generate")
        qubit_input_generator.generate.return_value = [Statevector([1, 0])]
        test_case_generator = TestCaseGenerator(PropertyTestExample, assessor_factor, qubit_input_generator)
        generated_test_case = test_case_generator.generate()

        assert generated_test_case.assessor == assessor_factor.build.return_value
        assert len(generated_test_case.circuit.qubits) == 1
        assert len(generated_test_case.circuit.data) == 1
        assert generated_test_case.num_experiments == 3
        assert generated_test_case.num_measurements == 2

    def test_generate_calls_correct_methods_when_correct_input(self, mocker: MockFixture):
        assessor_factor = mocker.patch.object(AssessorFactory, "build")
        assessor_factor.build.return_value = mocker.MagicMock()
        qubit_input_generator = mocker.patch.object(QubitInputGenerator, "generate")
        qubit_input_generator.generate.return_value = [Statevector([1, 0])]
        test_case_generator = TestCaseGenerator(PropertyTestExample, assessor_factor, qubit_input_generator)
        test_case_generator.generate()

        assessor_factor.build.assert_called_once()
        qubit_input_generator.generate.assert_called_once()
