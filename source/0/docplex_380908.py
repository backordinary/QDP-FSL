# https://github.com/SamirFarhat17/quantum-computer-programming-ibm/blob/eeb446026f480cdb48e4dc9c6d23b825300493c9/circuit-benchmarking/configs-benchmarks/benchmark/docplex.py
from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.problems import QuadraticProgram
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

class DocplexBenchmarks:

    version = 1
    params = ([-4,-3,-2,-1], [2, 3, 4, 5])
    param_names = ["lower bound", "upper bound"]


    def setup(self, lb, ub):
        docplex_model = Model('docplex')
        x = docplex_model.binary_var('x')
        y = docplex_model.integer_var(lb, ub, 'y')
        docplex_model.maximize(x * y)
        docplex_model.add_constraint(x <= y)

        qp = QuadraticProgram()
        qp.binary_var('x')
        qp.integer_var(name='y', lowerbound=lb, upperbound=ub)
        qp.maximize(quadratic={('x', 'y'): 1})
        qp.linear_constraint({'x': 1, 'y': -1}, '<=', 0)

        self._qup = qp

        seed = 671
        algorithm_globals.random_seed = seed
        qasm_sim = Aer.get_backend("aer_simulator")
        self._qins = QuantumInstance(
            backend=qasm_sim, shots=1000
        )
        if ub >= lb:
            self._docplex = Model('docplex')
            self._qp = from_docplex_mp(self._docplex)
        else:
            raise NotImplementedError

    @staticmethod
    def _generate_qubo(qp):
        model = Model('docplex')
        q_p = from_docplex_mp(model)
        return q_p

    def time_generate_qubo(self, _, __):
        self._generate_qubo(self._qp)

    def time_qaoa(self, _, __):
        CplexOptimizer().solve(self._qup)
