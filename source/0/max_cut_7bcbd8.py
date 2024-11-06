# https://github.com/SamirFarhat17/quantum-computer-programming-ibm/blob/eeb446026f480cdb48e4dc9c6d23b825300493c9/circuit-benchmarking/configs-benchmarks/benchmark/max_cut.py
from itertools import product
from timeit import timeit

import networkx as nx
from qiskit import Aer
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import algorithm_globals, QuantumInstance

from qiskit_optimization.algorithms import MinimumEigenOptimizer, GroverOptimizer
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo


class MaxcutBenchmarks:

    version = 1
    params = ([2, 4, 8, 12], [3, 5, 7, 9])
    param_names = ["number of nodes", "degree"]

    def setup(self, n, d):
        """setup"""
        seed = 123
        algorithm_globals.random_seed = seed
        qasm_sim = Aer.get_backend("qasm_simulator")
        self._qins = QuantumInstance(
            backend=qasm_sim, shots=1, seed_simulator=seed, seed_transpiler=seed
        )
        if n >= d:
            graph = nx.random_regular_graph(n=n, d=d)
            self._maxcut = Maxcut(graph=graph)
            self._qp = self._maxcut.to_quadratic_program()
        else:
            raise NotImplementedError

    @staticmethod
    def _generate_qubo(maxcut: Maxcut):
        q_p = maxcut.to_quadratic_program()
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(q_p)
        return qubo

    def time_generate_qubo(self, _, __):
        """generate time qubo"""
        self._generate_qubo(self._maxcut)

    def time_qaoa(self, _, __):
        """time qaoa"""
        meo = MinimumEigenOptimizer(
            min_eigen_solver=QAOA(optimizer=COBYLA(maxiter=1), quantum_instance=self._qins)
        )
        meo.solve(self._qp)

    def time_vqe(self, _, __):
        """time vqe"""
        meo = MinimumEigenOptimizer(
            min_eigen_solver=VQE(
                optimizer=COBYLA(maxiter=1), ansatz=EfficientSU2(), quantum_instance=self._qins
            )
        )
        meo.solve(self._qp)

    def time_grover(self, _, __):
        """time grover"""
        meo = GroverOptimizer(
            num_value_qubits=self._qp.get_num_vars() // 2,
            num_iterations=1,
            quantum_instance=self._qins,
        )
        meo.solve(self._qp)
