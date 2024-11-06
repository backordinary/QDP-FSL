# https://github.com/mgrzesiuk/qiskit-check/blob/74109e85011ded43230f66a62cdf0e9dabf0e1f4/tst/test_engine/test_runner/test_simulator_test_runner.py
from typing import Sequence, Union

from pytest_mock import MockFixture
from qiskit import QuantumCircuit
from scipy.spatial.transform import Rotation

from qiskit_check.test_engine.assessor import AssessorFactory
from qiskit_check.test_engine.concrete_property_test import ConcretePropertyTest
from qiskit_check.test_engine.generator import HaarInputGeneratorFactory, NaiveInputGeneratorFactory, \
    QubitInputGeneratorFactory
from qiskit_check.test_engine.test_runner import SimulatorTestRunner
from qiskit_check.property_test import PropertyTest
from qiskit_check.property_test.assertions import AbstractAssertion, AssertTransformedByProbability, AssertTransformedByState
from qiskit_check.property_test.resources import Qubit, AnyRange


class ExamplePropertyTest(PropertyTest):

    @property
    def circuit(self) -> QuantumCircuit:
        return QuantumCircuit(1)

    def get_qubits(self) -> Sequence[Qubit]:
        return [Qubit(AnyRange())]

    def assertions(self, qubits: Sequence[Qubit]) -> Union[AbstractAssertion, Sequence[AbstractAssertion]]:
        return [AssertTransformedByProbability(self.qubits[0], Rotation.identity())]

    @staticmethod
    def confidence_level() -> float:
        return 0.99

    @staticmethod
    def num_test_cases() -> int:
        return 3

    @staticmethod
    def num_measurements() -> int:
        return 100

    @staticmethod
    def num_experiments() -> int:
        return 100


class Example1PropertyTest(ExamplePropertyTest):
    def assertions(self, qubits: Sequence[Qubit]) -> Union[AbstractAssertion, Sequence[AbstractAssertion]]:
        return [AssertTransformedByState(self.qubits[0], 0, Rotation.identity())]


class Example2PropertyTest(ExamplePropertyTest):
    @staticmethod
    def num_measurements() -> int:
        return 1000

    @staticmethod
    def num_experiments() -> int:
        return 500


class ExampleFailPropertyTest(ExamplePropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.x(0)
        return qc


class TestSimulatorTestRunner:
    def test_run_tests_runs_all_tests_when_everything_correct(self, mocker: MockFixture):
        test_runner = SimulatorTestRunner("aer_simulator", mocker.MagicMock())
        assessor_factory = AssessorFactory()
        haar_factory = HaarInputGeneratorFactory()
        naive_factory = NaiveInputGeneratorFactory()
        tests = [
            ConcretePropertyTest(Example2PropertyTest, assessor_factory, naive_factory),
            ConcretePropertyTest(ExampleFailPropertyTest, assessor_factory, naive_factory),
            ConcretePropertyTest(Example1PropertyTest, assessor_factory, haar_factory)
        ]
        assert ["ExampleFailPropertyTest"], ["Example2PropertyTest", "Example1PropertyTest"] == test_runner.run_tests(tests)

    def test_run_tests_runs_all_tests_when_tomography_none(self, mocker: MockFixture):
        test_runner = SimulatorTestRunner("aer_simulator", mocker.MagicMock())
        assessor_factory = AssessorFactory()
        haar_factory = HaarInputGeneratorFactory()
        naive_factory = NaiveInputGeneratorFactory()
        tests = [
            ConcretePropertyTest(Example2PropertyTest, assessor_factory, naive_factory),
            ConcretePropertyTest(ExampleFailPropertyTest, assessor_factory, naive_factory),
            ConcretePropertyTest(Example1PropertyTest, assessor_factory, haar_factory)
        ]
        # example 1 property test should fail due to no tomography specified,o other tests should run correctly
        assert ["ExampleFailPropertyTest", "Example1PropertyTest"], ["Example2PropertyTest"] == test_runner.run_tests(tests)
