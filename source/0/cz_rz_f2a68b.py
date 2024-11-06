# https://github.com/georgios-ts/qc-mentorship/blob/8e930b6a3ca6c31b5b947dc0ab58f0fe87715c7e/task_3/optimization/commutation/cz_rz.py
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library.standard_gates import CZGate, RZGate



class CZRZCommutation(TransformationPass):
    """
    If back is set to True,
    apply the following identity to enable other optimizations.
            input:
                q_0: ──────────■─
                     ┌───────┐ │
                q_1: ┤ RZ(φ) ├─■─
                     └───────┘
            output:
                q_0: ─■──────────
                      │ ┌───────┐
                q_1: ─■─┤ RZ(φ) ├
                        └───────┘
    If it is set to False,
    the reverse transformation is applied.
    """

    def __init__(self, back):
        super().__init__()
        self.back = back

    def run(self, dag):

        for cz_node in dag.named_nodes('cz'):
            if self.back:
                nodes = filter(lambda x: x.name == 'rz',
                               dag.predecessors(cz_node))
            else:
                nodes = filter(lambda x: x.name == 'rz',
                               dag.successors(cz_node))

            angles = []
            qubits = []
            ctrl_index = cz_node.qargs[0].index
            for rz_node in nodes:
                angles.append(rz_node.op.params[0])
                if rz_node.qargs[0].index == ctrl_index:
                    qubits.append(0)
                else:
                    qubits.append(1)

                dag.remove_op_node(rz_node)

            dag.substitute_node_with_dag(cz_node,
                                         _dag(angles, qubits, self.back))

        return dag


def _dag(angles, qubits, back=True):
    """ Return a new DAGCircuit with a CZ gate and RZ gates applied (front or back). """
    qarg = QuantumRegister(2)

    dag = DAGCircuit()
    dag.add_qreg(qarg)

    cz_node = dag.apply_operation_back(CZGate(),
                                    [qarg[0], qarg[1]])

    for angle, qubit in zip(angles, qubits):
        if back:
            dag.apply_operation_back(RZGate(angle),
                                    [qarg[qubit]], [])
        else:
            dag.apply_operation_front(RZGate(angle),
                                    [qarg[qubit]], [])

    return dag
