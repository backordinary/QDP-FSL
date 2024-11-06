# https://github.com/mgrzesiuk/qiskit-check/blob/74109e85011ded43230f66a62cdf0e9dabf0e1f4/qiskit_check/test_engine/assessor.py
from typing import Dict, List, Tuple, Set, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.result import Result
from qiskit_utils import parse_result, parse_counts

from qiskit_check.test_engine.measurement_location import MeasurementLocation
from qiskit_check.test_engine.p_value_correction import AbstractCorrection
from qiskit_check.property_test.assertions.abstract_assertion import AbstractAssertion
from qiskit_check.property_test.property_test import PropertyTest
from qiskit_check.property_test.property_test_errors import IncorrectAssertionError
from qiskit_check.property_test.resources.test_resource import Qubit, ConcreteQubit
from qiskit_check.property_test.test_results import TestResult


class Assessor:
    """
    class used for managing and evaluating assertions for each property test case
    """
    def __init__(
            self, assertions: Sequence[AbstractAssertion], confidence_level: float,
            resource_matcher: Dict[Qubit, ConcreteQubit], measurement_locations: Dict[Tuple[Qubit], List[MeasurementLocation]]) -> None:
        """
        initialize
        Args:
            assertions: sequence of assertions specified
            confidence_level: confidence level
            resource_matcher: mapping between specified qubits and a generated qubit object containing info of
            qubit index in the circuit and its initial value
        """
        self.assertions = assertions
        self.resource_matcher = resource_matcher
        self.confidence_level = confidence_level
        self.measurement_locations = measurement_locations

    def assess(self, results: List[Dict[str, Tuple[Result, QuantumCircuit]]], corrector: AbstractCorrection, num_measurements: int, num_experiments: int) -> None:
        """
        evaluate assertions given test results
        Args:
            results: test results from test runner
            corrector: p-value correction object
            num_measurements: number of measurements per experiment
            num_experiments: number of experiments done

        Returns: None, AssertError thrown if assertion fails

        """
        for assertion in self.assertions:
            assertion_input = TestResult(self._get_result(results, assertion), self._get_counts(results, assertion))
            p_value = assertion.get_p_value(assertion_input, self.resource_matcher, num_measurements, num_experiments)
            confidence_level = corrector.get_corrected_confidence_level()
            assertion.verify(confidence_level, p_value)

    def _get_counts(self, experiment_results: List[Dict[str, Tuple[Result, QuantumCircuit]]], assertion: AbstractAssertion) -> List[List[Dict[str, int]]]:
        measurement_names = self.get_measurement_names(assertion)

        # TODO: this can be optimized
        qubits = assertion.get_qubits()
        results_per_instruction = []
        for instruction in assertion.measurements:
            encoding = self.encode_measurement(qubits, assertion.location, instruction)
            results = []
            for experiment_result in experiment_results:
                results.append(parse_counts(*experiment_result[encoding], measurement_names=measurement_names))
            results_per_instruction.append(results)
        return results_per_instruction

    
    def _get_result(self, experiment_results: List[Dict[str, Tuple[Result, QuantumCircuit]]], assertion: AbstractAssertion) -> Dict[Qubit, List[List[float]]]:
        measurement_names = self.get_measurement_names(assertion)
        # TODO: this can be optimized
        parsed_result = {}
        qubits = assertion.get_qubits()
        for qubit in qubits:
            results_per_instruction = []
            for instruction in assertion.measurements:
                encoding = self.encode_measurement(qubits, assertion.location, instruction)
                results = []
                for experiment_result in experiment_results:
                    results.append(parse_result(*experiment_result[encoding], measurement_names=measurement_names)[self.resource_matcher[qubit].qubit_index])
                results_per_instruction.append(results)
            parsed_result[qubit] = assertion.combiner(results_per_instruction)
        return parsed_result

    @staticmethod
    def encode_measurement(qubits: Tuple[Qubit], location: int, instruction: Instruction) -> str:
        return f"{hash(qubits)}-{location}-{instruction.name}"
    
    @staticmethod
    def get_measurement_names(assertion: AbstractAssertion) -> Set[str]:
        measurement_names = set()
        for measurement in assertion.measurements:
            measurement_names.add(measurement.name)
        return measurement_names


class AssessorFactory:
    @staticmethod
    def build(property_test: PropertyTest, resource_matcher: Dict[Qubit, ConcreteQubit]) -> Assessor:
        """
        builds an assessor given property test and resource matcher
        Args:
            property_test: property test object (1 test case)
            resource_matcher: mapping between specified qubits and a generated qubit object containing info of
            qubit index in the circuit and its initial value

        Returns: Assessor set up with the given parameters

        """
        test_assertions = property_test.assertions(property_test.qubits)
        if isinstance(test_assertions, Sequence):
            for assertion in test_assertions:
                if not isinstance(assertion, AbstractAssertion):
                    raise IncorrectAssertionError(property_test)
            assertions = test_assertions
        elif isinstance(test_assertions, AbstractAssertion):
            assertions = (test_assertions, )
        else:
            raise IncorrectAssertionError(property_test)
        
        measurement_locations = AssessorFactory._create_measurement_locations(assertions)
        
        return Assessor(assertions, property_test.confidence_level(), resource_matcher, measurement_locations)
    
    @staticmethod
    def _create_measurement_locations(assertions: Sequence[AbstractAssertion]) -> Dict[Tuple[Qubit], List[MeasurementLocation]]:
        measurement_locations = {}
        
        for assertion in assertions:
            qubits = assertion.get_qubits()
            for instruction in assertion.measurements:
                if qubits in measurement_locations:
                    measurement_locations[qubits].append(MeasurementLocation(assertion.location, instruction))
                else:
                    measurement_locations[qubits] = [MeasurementLocation(assertion.location, instruction)]

        return measurement_locations
