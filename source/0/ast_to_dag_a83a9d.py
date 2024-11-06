# https://github.com/qis-unipr/dqc-circuit/blob/cc887a47f2f52a08229f24e9b40661b296cb431e/dqc_parser/ast_to_dag.py
from multiprocessing import Barrier

from qiskit import QuantumRegister, ClassicalRegister, QiskitError
from qiskit.circuit import Reset
from qiskit.converters.ast_to_dag import AstInterpreter
from qiskit.dagcircuit import DAGCircuit

from distributed_circuit.instructions import EPRInstr, RemoteCxInstr, EntSwapInstr


def ast_to_dag(ast):
    """Build a ``DAGCircuit`` object from an AST ``Node`` object.

    Args:
        ast (Program): a Program Node of an AST (dqc_parser's output)

    Return:
        DAGCircuit: the DAG representing an OpenQASM's AST

    Raises:
        QiskitError: if the AST is malformed.

    Example:
        .. jupyter-execute::

            from qiskit.converters import ast_to_dag
            from qiskit import qasm, QuantumCircuit, ClassicalRegister, QuantumRegister
            from qiskit.visualization import dag_drawer
            %matplotlib inline

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            qasm_str = circ.qasm()
            ast = qasm.Qasm(data=qasm_str).parse()
            dag = ast_to_dag(ast)
            dag_drawer(dag)
    """
    dag = DAGCircuit()
    DQCAstInterpreter(dag)._process_node(ast)

    return dag


class DQCAstInterpreter(AstInterpreter):

    def __init__(self, dag):
        super().__init__(dag)

    def _process_node(self, node):
        """Carry out the action associated with a node."""
        if node.type == "program":
            self._process_children(node)

        elif node.type == "qreg":
            qreg = QuantumRegister(node.index, node.name)
            self.dag.add_qreg(qreg)

        elif node.type == "creg":
            creg = ClassicalRegister(node.index, node.name)
            self.dag.add_creg(creg)

        elif node.type == "id":
            raise QiskitError("internal error: _process_node on id")

        elif node.type == "int":
            raise QiskitError("internal error: _process_node on int")

        elif node.type == "real":
            raise QiskitError("internal error: _process_node on real")

        elif node.type == "indexed_id":
            raise QiskitError("internal error: _process_node on indexed_id")

        elif node.type == "id_list":
            # We process id_list nodes when they are leaves of barriers.
            return [self._process_bit_id(node_children) for node_children in node.children]

        elif node.type == "primary_list":
            # We should only be called for a barrier.
            return [self._process_bit_id(m) for m in node.children]

        elif node.type == "gate":
            self._process_gate(node)

        elif node.type == "custom_unitary":
            self._process_custom_unitary(node)

        elif node.type == "universal_unitary":
            self._process_u(node)

        elif node.type == "cnot":
            self._process_cnot(node)

        elif node.type == "expression_list":
            return node.children

        elif node.type == "binop":
            raise QiskitError("internal error: _process_node on binop")

        elif node.type == "prefix":
            raise QiskitError("internal error: _process_node on prefix")

        elif node.type == "measure":
            self._process_measure(node)

        elif node.type == "format":
            self.version = node.version()

        elif node.type == "barrier":
            ids = self._process_node(node.children[0])
            qubits = []
            for qubit in ids:
                for j, _ in enumerate(qubit):
                    qubits.append(qubit[j])
            self.dag.apply_operation_back(Barrier(len(qubits)), qubits, [])

        elif node.type == "entswap":
            self._process_entswap(node)

        elif node.type == "remoteCx":
            self._process_remotecx(node)

        elif node.type == "epr":
            self._process_epr(node)

        elif node.type == "reset":
            id0 = self._process_bit_id(node.children[0])
            for i, _ in enumerate(id0):
                reset = Reset()
                reset.condition = self.condition
                self.dag.apply_operation_back(reset, [id0[i]], [])

        elif node.type == "if":
            self._process_if(node)

        elif node.type == "opaque":
            self._process_gate(node, opaque=True)

        elif node.type == "external":
            raise QiskitError("internal error: _process_node on external")

        else:
            raise QiskitError(
                "internal error: undefined node type",
                node.type,
                "line=%s" % node.line,
                "file=%s" % node.file,
            )
        return None

    def _process_epr(self, node):
        """Process a EPR gate node."""
        id0 = self._process_bit_id(node.children[0])
        id1 = self._process_bit_id(node.children[1])
        if not (len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1):
            raise QiskitError("internal error: qreg size mismatch",
                              "line=%s" % node.line, "file=%s" % node.file)

        self.dag.apply_operation_back(EPRInstr(), [id0[0], id1[0]])

    def _process_remotecx(self, node):
        """Process a REMOTECX gate node."""
        id0 = self._process_bit_id(node.children[0])
        eid0 = self._process_bit_id(node.children[1])
        eid1 = self._process_bit_id(node.children[2])
        id1 = self._process_bit_id(node.children[3])
        if not (len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1):
            raise QiskitError("internal error: qreg size mismatch",
                              "line=%s" % node.line, "file=%s" % node.file)
        if not (len(eid0) == len(eid1) or len(eid0) == 1 or len(eid1) == 1):
            raise QiskitError("internal error: qreg size mismatch",
                              "line=%s" % node.line, "file=%s" % node.file)

        self.dag.apply_operation_back(RemoteCxInstr(), [id0[0], eid0[0], eid1[0], id1[0]])

    def _process_entswap(self, node):
        """Process a ENTSWAP gate node."""
        ids = self._process_node(node.children[0])
        qubits = []
        for qubit in ids:
            for j, _ in enumerate(qubit):
                qubits.append(qubit[j])
        # cbits = []
        # for cbit in self._process_node(node.children[1]):
        #     for j, _ in enumerate(cbit):
        #         cbits.append(cbit[j])
        # self.dag.apply_operation_back(EntSwapInstr(len(qubits)), qubits, cbits)
        self.dag.apply_operation_back(EntSwapInstr(len(qubits)), qubits)
