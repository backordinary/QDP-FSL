# https://github.com/mgrzesiuk/qiskit-check/blob/6c81ce8075291e51434f7ba71ef15710343d9874/case_studies/grover_search/tst.py
from abc import ABC
from typing import Collection, Dict, List, Sequence

from qiskit import QuantumCircuit

from case_studies.example_test_base import ExampleTestBase
from case_studies.grover_search.src import grover_search, oracle
from qiskit_check.property_test.assertions import AbstractAssertion, AssertTrue
from qiskit_check.property_test.resources.test_resource import ConcreteQubit, Qubit
from qiskit_check.property_test.resources.qubit_range import QubitRange


class AbstractGroverSearchPropertyTest(ExampleTestBase, ABC):
    def get_qubits(self) -> Collection[Qubit]:
        return [Qubit(QubitRange(0, 0, 0, 0)) for _ in range(5)]

    def assertions(self, qubits: Sequence[Qubit]) -> AbstractAssertion:
        return AssertTrue(qubits, self.most_probable, 1)
    
    def most_probable(self, measurement: List[Dict[str, int]], resource_matcher: Dict[Qubit, ConcreteQubit]) -> float:
        max_counts = 0
        desired_state_counts = 0
        for state, counts in measurement[0].items():
            if counts > max_counts:
                max_counts = counts
            if state == "1"*len(self.qubits):
                desired_state_counts = counts
        return int(desired_state_counts == max_counts)

class GroverSearchPropertyTest(AbstractGroverSearchPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qubit_number = len(self.qubits)
        return grover_search(1, oracle(QuantumCircuit(qubit_number)))
