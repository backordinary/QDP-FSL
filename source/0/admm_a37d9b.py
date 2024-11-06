# https://github.com/sergiogh/tsp_mayhem/blob/2137f2926cde55771176643a0218bb899805b031/solvers/admm.py
import numpy as np
from qiskit import Aer, IBMQ
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, QAOA
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit.optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit.optimization.applications.ising import tsp
from qiskit.optimization.applications.ising.common import sample_most_likely


class Admm:

    def calculate(self, G, cost_matrix, starting_node = 0):

        # Create nodes array for the TSP solver in Qiskit
        coords = []
        for node in G.nodes:
            coords.append(G.nodes[node]['pos'])

        tsp_instance = tsp.TspData(name = "TSP", dim = len(G.nodes), coord = coords, w = cost_matrix)

        qubitOp, offset = tsp.get_operator(tsp_instance)
        print("Qubits needed: ", qubitOp.num_qubits)
        #print(qubitOp.print_details())

        #backend = Aer.get_backend('statevector_simulator')
        backend = Aer.get_backend('qasm_simulator')

        # Create QUBO based on qubitOp from the TSP
        qp = QuadraticProgram()
        qp.from_ising(qubitOp, offset, linear=True)

        admm_params = ADMMParameters(
                            rho_initial=1001,
                            beta=1000,
                            factor_c=900,
                            maxiter=100,
                            three_block=True, tol=1.e-6)

        qubo_optimizer = MinimumEigenOptimizer(QAOA(quantum_instance=backend))
        convex_optimizer = CobylaOptimizer()
        admm = ADMMOptimizer(params=admm_params,
                             qubo_optimizer=qubo_optimizer,
                             continuous_optimizer=convex_optimizer)

        quantum_instance = QuantumInstance(backend)

        result = admm.solve(qp)
        print(result)

        x = sample_most_likely(result.eigenstate)

        if(tsp.tsp_feasible(x)):
            z = tsp.get_tsp_solution(x)
            print('solution:', z)
            return z
        else:
            print('no solution:', x)
            return []


    def iterations(_eval_count, param_set, means, estimator_error):
        print(_eval_count)
        print(param_set)
        print(means)
        print(estimator_error)
