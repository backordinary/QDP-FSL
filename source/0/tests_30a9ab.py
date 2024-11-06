# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/MTF/tests.py
from mutation import Mutant
from qiskit import QuantumCircuit
from MR import MR
from typing import List
TESTS = 100

class QuantumTest:
    def __init__(self, qc, tests = TESTS):
        self.qc = qc
        self.tests = tests

    def test(self) -> bool:
        pass
    
    def construct_circuit(self) -> QuantumCircuit:
        return None


class MetamorficTest(QuantumTest):
    def __init__(self, MR: List[MR], qc: QuantumCircuit = None, tests = TESTS):
        super().__init__(qc, tests)
        if (self.qc is None):
            self.qc = self.construct_circuit()
        self.MR = MR

    def test(self, qc) -> bool:
        success = True
        for mr in self.MR:
            success = success and mr(qc)
        return success

    def mutation_score(self) -> float:
        success = 0.0
        for _ in range(self.tests):
            mutant = Mutant(self.qc)
            success += self.test(mutant.qc)
        return 1 - (success /self.tests)

    def run(self) -> float:
        success = 0.0
        for _ in range(self.tests):
            success += self.test(self.qc)
        return success / self.tests