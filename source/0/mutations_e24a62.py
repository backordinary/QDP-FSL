# https://github.com/mgrzesiuk/qiskit-check/blob/f06df70750eb58b685825aa403c58b5675dcbe75/e2e_test/entangled_assertion_test/mutations.py
from qiskit import QuantumCircuit

from e2e_test.entangled_assertion_test.tst import AbstractEntanglePropertyPropertyTest


class MutationNoHEntanglePropertyTest(AbstractEntanglePropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.measure_all()
        return qc


class MutationNoCXEntanglePropertyTest(AbstractEntanglePropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()
        return qc


class MutationAdditionalCXGateEntanglePropertyTest(AbstractEntanglePropertyPropertyTest):
    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.measure_all()
        return qc
