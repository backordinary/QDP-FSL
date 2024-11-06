# https://github.com/Zshan0/classiq-22/blob/f9d3418f81004f33a107a607719a1da4f31eb7ce/src/circuit_optimizer.py
from qiskit import QuantumCircuit
from lietrotter import Lie
from constructor import hamiltonian_circuit_error
from parser import export_circuit

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGatesSimpleCommutation,
    CommutativeCancellation,
    CXCancellation,
    CommutationAnalysis,
)


class Optimizer:
    def __init__(self, basis_gates=None):
        if basis_gates is None:
            basis_gates = ["u3", "cx"]
        self.basis_gates = basis_gates

        simple_commutation_pass = Optimize1qGatesSimpleCommutation(basis=basis_gates)
        cancel_cx_pass = CXCancellation()
        commutative_cancellation_pass = CommutativeCancellation(basis_gates=basis_gates)
        commutative_analysis_pass = CommutationAnalysis()

        self.passes = [
            simple_commutation_pass,
            cancel_cx_pass,
            commutative_analysis_pass,
            commutative_cancellation_pass,
        ]
        self.manager = PassManager(self.passes)

    @staticmethod
    def _pass_call_back(original_depth, dict_):
        print(f"{dict_['pass_']} completed in {dict_['time']}")
        print(f"original depth: {original_depth} new depth: {dict_['dag'].depth()}")

    @staticmethod
    def _end_call_back(**kwargs):
        print(f'new:{kwargs["new_depth"]} old:{kwargs["original_depth"]}')

    def transipile_optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circ = circuit.copy()
        original_depth = circ.depth()

        def call_back(**kwargs):
            self._pass_call_back(original_depth, kwargs)

        transpiled_circ = self.manager.run(circ, callback=call_back)

        assert (
            type(transpiled_circ) is QuantumCircuit
        ), "transpile returned list, not supported currently."
        new_depth = transpiled_circ.depth()
        self._end_call_back(new_depth=new_depth, original_depth=original_depth)

        return transpiled_circ

    def __call__(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return self.transipile_optimize(circuit)


def main():
    hamiltonian = "H2"
    constructor = Lie(1)
    constructor.load_hamiltonian(hamiltonian)
    constructor.get_circuit()
    pauli_op = constructor.pauli_op
    dec_circuit = constructor.decompose_circuit()

    optimizer = Optimizer()
    optim_circ = optimizer.transipile_optimize(dec_circuit)
    new_error = hamiltonian_circuit_error(optim_circ, pauli_op)
    original_error = hamiltonian_circuit_error(dec_circuit, pauli_op)

    print(f"original error:{original_error}, new error:{new_error}")
    export_circuit(
        optim_circ,
        f"circuits/{constructor.hamiltonian}_{constructor.method}_{optim_circ.depth()}.qasm",
    )


if __name__ == "__main__":
    main()
