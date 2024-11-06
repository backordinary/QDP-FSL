# https://github.com/rkruegs123/qcircuit-optimization/blob/bc9d4668b73960ba0c792249e08d327138318389/baseline_optimizers.py
import pdb
import sys
sys.path.append('../pyzx')
import pyzx as zx

# pytket
from pytket.extensions.pyzx import pyzx_to_tk, tk_to_pyzx
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, RepeatWithMetricPass, \
    SequencePass, FullPeepholeOptimise, RebasePyZX
from pytket.qasm import circuit_to_qasm_str

# qiskit
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager

from base_optimizer import Optimizer


class QiskitTranspiler(Optimizer):

    def __init__(self, opt_level):
        self.opt_level = opt_level
        super().__init__()

    def _optimize(self, c):
        qasm_str = c.to_qasm()
        qiskit_circ = QuantumCircuit.from_qasm_str(qasm_str)
        qiskit_circ_opt = transpile(qiskit_circ, optimization_level=self.opt_level)
        opt_qasm_str = qiskit_circ_opt.qasm()
        c_opt = zx.Circuit.from_qasm(opt_qasm_str)
        return c_opt

    @property
    def name(self):
        return f"transpile\n(qiskit, level={self.opt_level})"

    @property
    def desc(self):
        return "Qiskit's default transpiler. Makes use of various individual Transpiler Passes."

class TketBasic(Optimizer):
    def _optimize(self, c):
        c_tket = pyzx_to_tk(c)
        RemoveRedundancies().apply(c_tket)
        c_opt = tk_to_pyzx(c_tket)
        return c_opt

    @property
    def name(self):
        return f"RemoveRedundancies\n(tket)"

    @property
    def desc(self):
        return "Tket's most basic optimization method"

class TketMinCX(Optimizer):
    def _optimize(self, c):
        c_tket = pyzx_to_tk(c)
        cost = lambda c : c.n_gates_of_type(OpType.CX)
        comp = RepeatWithMetricPass(SequencePass([CommuteThroughMultis(), RemoveRedundancies()]), cost)
        comp.apply(c_tket)
        c_opt = tk_to_pyzx(c_tket)
        return c_opt

    @property
    def name(self):
        return f"RepeatPass\n(tket)"

    @property
    def desc(self):
        return "Example from tket tutorial that minimizes number of CNOT gates via a combinator. For more info, see https://cqcl.github.io/pytket/build/html/manual_compiler.html#combinators"

class TketFullPeephole(Optimizer):
    def _optimize(self, c):
        c_tket = pyzx_to_tk(c)
        FullPeepholeOptimise().apply(c_tket)

        # FIXME: Failing equality!
        # RebasePyZX().apply(c_tket)
        # c_opt = tk_to_pyzx(c_tket)

        # FIXME: Failing equality!
        qasm_str = circuit_to_qasm_str(c_tket)
        c_opt = zx.Circuit.from_qasm(qasm_str)

        return c_opt

    @property
    def name(self):
        return f"FullPeepholeOptimise\n(tket)"

    @property
    def desc(self):
        return "Tket's one-size-fits-all optimization method. Note that we do not target a particular backend. For more info, see https://cqcl.github.io/pytket/build/html/manual_compiler.html#predefined-sequences"

class FullOptimize(Optimizer):

    def _optimize(self, c):
        return zx.full_optimize(c)

    @property
    def name(self):
        return f"full_optimize\n(PyZX)"

    @property
    def desc(self):
        return "PyZX's most general circuit optimization procedure"

class FullReduce(Optimizer):

    def _optimize(self, c):
        g = c.to_graph()
        zx.full_reduce(g, quiet=True)
        c_opt = zx.extract_circuit(g.copy()) # FIXME: maybe don't need g.copy()?
        return c_opt

    @property
    def name(self):
        return f"full_reduce\n(PyZX)"

    @property
    def desc(self):
        return "PyZX's most general graph optimization procedure"

class TeleportReduce(Optimizer):

    def _optimize(self, c):
        g = c.to_graph()
        zx.teleport_reduce(g)
        c_opt = zx.Circuit.from_graph(g)
        return c_opt

    @property
    def name(self):
        return f"teleport_reduce\n(PyZX)"

    @property
    def desc(self):
        return "Variant of full_reduce that only moves around phases and preserves circuit structure"
