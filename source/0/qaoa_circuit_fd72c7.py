# https://github.com/TDoGoodT/generic_qaoa/blob/55fa5586fb4282f31a6f548afe398fa072fc049b/generic_qaoa/qaoa/qaoa_circuit.py
from typing import List

from sympy import Symbol, Add, Mul
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..utils import z_to_qubit_index


class QaoaCircuitFactory(object):
    @staticmethod
    def create_circuit(clauses: List[Symbol], betas, gammas, p, n_qbits):
        qubits_indexes = range(n_qbits)
        qc = QaoaCircuitFactory.QaoaCircuit(n_qbits, betas=betas, gammas=gammas)
        qc.add_hadamard(qubits_indexes)
        for i in range(p):
            qc.add_prepare_gate(clauses, gammas[i])
            qc.barrier()
            qc.add_mix_gate(betas[i], qubits_indexes)
        qc.measure_all()
        return qc

    @staticmethod
    def create_parameterized_circuit(clauses: List[Symbol], p, n_qbits):
        qubits_indexes = range(n_qbits)
        gammas, betas = ParameterVector(name="gammas", length=p), ParameterVector(name="betas", length=p)
        qc = QaoaCircuitFactory.QaoaCircuit(n_qbits, betas=betas, gammas=gammas)
        qc.add_hadamard(qubits_indexes)
        for i in range(p):
            qc.add_prepare_gate(clauses, gammas[i])
            qc.barrier()
            qc.add_mix_gate(betas[i], qubits_indexes)
        qc.measure_all()
        return qc

    class QaoaCircuit(QuantumCircuit):
        def __init__(self, *regs, betas, gammas):
            super().__init__(*regs)
            self.betas = betas
            self.gammas = gammas

        def _append_clause_to_circuit(self, expanded_H, angle):
            args = Add.make_args(expanded_H)
            for _arg in args:
                self._add_z_rotation(_arg, angle)

        def _add_cnot(self, control, target):
            control_index, target_index = z_to_qubit_index(control), z_to_qubit_index(target)
            self.cnot(control_index, target_index)

        def _add_barrier(self, terms):
            self.barrier([z_to_qubit_index(term) for term in terms])

        def _add_z_rotation(self, _arg, angle):
            if _arg.is_number:
                return
            terms = Mul.make_args(_arg)
            _terms, angle = (terms[1:], float(terms[0]) * angle) if terms[0].is_number else (terms, angle)
            if len(_terms) == 1:
                self.rz(angle, z_to_qubit_index(_terms[0]))
                return
            self._add_barrier(_terms)
            for i in range(len(_terms) - 1):
                self._add_cnot(_terms[i], _terms[i + 1])
            self.rz(angle, z_to_qubit_index(_terms[-1]))
            for i in reversed(range(len(_terms) - 1)):
                self._add_cnot(_terms[i], _terms[i + 1])
            self._add_barrier(_terms)

        def add_hadamard(self, qubits_indexes):
            for q_idx in qubits_indexes:
                self.h(q_idx)

        def add_prepare_gate(self, clauses, angle):
            for clause in clauses:
                self._append_clause_to_circuit(clause.hamiltonian, angle)

        def add_mix_gate(self, angle, qubits_indexes):
            for q_idx in qubits_indexes:
                self.rx(2 * angle, q_idx)
