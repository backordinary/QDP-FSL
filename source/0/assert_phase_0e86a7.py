# https://github.com/mgrzesiuk/qiskit-check/blob/74109e85011ded43230f66a62cdf0e9dabf0e1f4/qiskit_check/property_test/assertions/assert_phase.py
from math import sqrt, atan2, pi, isclose
from typing import Dict, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from scipy.stats import ttest_1samp

from qiskit_check.property_test.assertions import AbstractAssertion
from qiskit_check.property_test.test_results.test_result import TestResult

from qiskit_check.property_test.resources import Qubit, ConcreteQubit


class AssertPhase(AbstractAssertion):
    """
    assert phase of a qubit
    method: https://quantum-computing.ibm.com/composer/docs/iqx/guide/introducing-qubit-phase
    WARNING: the system has to be in superposition before asserting phase, that is start from state 0 and then h gate has to be applied at the
    beginning, after that one can apply operations which change phase and that phase will be evaluated
    """
    def __init__(
            self, qubit: Qubit, expected_phases: float, location: int = None) -> None:
        """

        Args:
            qubit: qubit to measure
            expected_phases: expected value of phase (in radians)
            location: where to measure in circuit
        """
        super().__init__(self.get_xy_measurements(), location, self.combiner)
        self.qubit = qubit
        self.expected_phases = expected_phases

    @staticmethod
    def get_ramsey_x_measurement() -> Instruction:
        measure_name = "measure_x"
        qc = QuantumCircuit(1, 1, name=measure_name)
        qc.h(0)
        qc.measure(0, 0)
        return qc.to_instruction(label=measure_name)

    @staticmethod
    def get_ramsey_y_measurement() -> Instruction:
        measure_name = "measure_y"
        qc = QuantumCircuit(1, 1, name=measure_name)
        qc.sdg(0)
        qc.h(0)
        qc.measure(0, 0)
        return qc.to_instruction(label=measure_name)

    @staticmethod
    def get_xy_measurements():
        return [AssertPhase.get_ramsey_x_measurement(), AssertPhase.get_ramsey_y_measurement()]

    @staticmethod
    def combiner(experiments: List[List[Dict[str, int]]]) -> List[List[float]]:
        results = [] #list of x phase, y phase, z phase for each test run

        for i in range(len(experiments[0])):
            p0x = experiments[0][i]["0"]/(experiments[0][i]["0"] + experiments[0][i]["1"])
            p1x = experiments[0][i]["1"]/(experiments[0][i]["0"] + experiments[0][i]["1"])
            dx = p0x - p1x

            p0y = experiments[1][i]["0"]/(experiments[1][i]["0"] + experiments[1][i]["1"])
            p1y = experiments[1][i]["1"]/(experiments[1][i]["0"] + experiments[1][i]["1"])
            dy = p0y - p1y


            norm = dx**2 + dy**2
            if norm == 0:
                expected_phase = np.NaN
            else:
                x = dx/sqrt(norm)
                y = dy/sqrt(norm)
                expected_phase = atan2(y, x)
            results.append(expected_phase)

        return [results]
    def get_p_value(self, experiments: TestResult, resource_matcher: Dict[Qubit, ConcreteQubit], num_measurements: int, num_experiments: int) -> float:
        return ttest_1samp(experiments.individual_measurements[self.qubit][0], self.expected_phases, alternative="two-sided").pvalue
