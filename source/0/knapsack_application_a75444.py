# https://github.com/SamirFarhat17/quantum-computer-programming-ibm/blob/eeb446026f480cdb48e4dc9c6d23b825300493c9/circuit-benchmarking/configs-benchmarks/benchmark/knapsack_application.py
import random
from qiskit import Aer
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer, GroverOptimizer
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.converters import QuadraticProgramToQubo


class KnapsackBenchmarks:
    params = ([2, 3, 4, 5], [2, 4, 8, 16])
    param_names = ["number of items", "max_weights"]

    def setup(self, num_items, max_weights):
        """setup"""
        seed = 10
        algorithm_globals.random_seed = seed
        qasm_sim = Aer.get_backend("aer_simulator")
        self._qins = QuantumInstance(
            backend=qasm_sim, shots=1, seed_simulator=seed, seed_transpiler=seed
        )
        random.seed(seed)
        values = [random.randint(1, max_weights) for _ in range(num_items)]
        weights = [random.randint(1, max_weights) for _ in range(num_items)]
        self._knapsack = Knapsack(values, weights, max_weights)
        self._qp = self._knapsack.to_quadratic_program()

    @staticmethod
    def _generate_qubo(knapsack: Knapsack):
        q_p = knapsack.to_quadratic_program()
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(q_p)
        return qubo

    def time_generate_qubo(self, _, __):
        self._generate_qubo(self._knapsack)

    def time_qaoa(self, _, __):
        meo = MinimumEigenOptimizer(
            min_eigen_solver=QAOA(optimizer=COBYLA(maxiter=1), quantum_instance=self._qins)
        )
        meo.solve(self._qp)

    def time_vqe(self, _, __):
        meo = MinimumEigenOptimizer(
            min_eigen_solver=VQE(
                optimizer=COBYLA(maxiter=1), ansatz=EfficientSU2(),
                quantum_instance=self._qins
            )
        )
        meo.solve(self._qp)

    def time_grover(self, _, __):
        meo = GroverOptimizer(
            num_value_qubits=self._qp.get_num_vars(),
            num_iterations=1,
            quantum_instance=self._qins,
        )
        meo.solve(self._qp)
