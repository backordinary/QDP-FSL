# https://github.com/Namr/YAQCS/blob/65a286f05fc917a39462d23ab1ad368f84466e7c/BQSKit%20Extensions/bqskit2qiskit.py
from qiskit import QuantumCircuit
from bqskit import Circuit
from bqskit import UnitaryMatrix
from bqskit.ir.gates import RXXGate, RXGate, RYGate, RZGate, CircuitGate, U3Gate
from qiskit.circuit.library.generalized_gates import GMS


def bqskit2qiskit(bqCirc: Circuit):
    qqc = QuantumCircuit(bqCirc.num_qudits)

    def transform_location(location):
        return bqCirc.num_qudits - location - 1

    for op in list(bqCirc.operations()):
        if op.gate.name == "U3Gate":
            qqc.u(op.params[0], op.params[1], op.params[2],
                  transform_location(op.location[0]))
        if op.gate.name == "RXXGate":
            qqc.rxx(op.params[0], transform_location(op.location[0]),
                    transform_location(op.location[1]))
        if op.gate.name == "RXXXGate":
            xxx = GMS(3, (op.params[0], op.params[0], op.params[0]))
            qqc.append(xxx, (transform_location(
                op.location[0]), transform_location(
                    op.location[1]), transform_location(op.location[2])))

    qqc.save_unitary()
    return qqc
