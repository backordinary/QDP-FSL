# https://github.com/donghufeng/mq_benchmark/blob/cc293a91306d9653ed60b567c1e1ac4329d99531/qaoa/benchmark_qiskit.py
from qiskit import Aer
from qiskit_optimization.applications import Maxcut
from qiskit.utils import algorithm_globals
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
import numpy as np


def bench(hyperparams={}):
    """
    Performs QAOA optimizations.

    Args:
            hyperparams (dict): hyperparameters to configure this benchmark

                    * 'graph': Graph represented as a NetworkX Graph class

                    * 'n_layers': Number of layers in the QAOA circuit

                    * 'shots': The number of samples.
    """

    graph = hyperparams['graph']
    n_layers = hyperparams['n_layers']
    shots = hyperparams['shots']

    n_qubits = len(graph.nodes)
    edges = list(graph.edges) 

    # Computing the weight matrix from the random graph
    adjacent_matrix = np.zeros([n_qubits, n_qubits])
    for u, v in edges:
        adjacent_matrix[u][v] += 1
    # print(adjacent_matrix)

    # Mapping to the Ising problem
    max_cut = Maxcut(adjacent_matrix)
    qp = max_cut.to_quadratic_program()
    print(qp.export_as_lp_string())

    # get the corresponding Ising Hamiltonian
    qubit_op, offset = qp.to_ising()
    # print("Offset:", offset)
    # print("Ising Hamiltonian:\n_qubits", str(qubit_op))

    algorithm_globals.random_seed = 10598

    optimizer = COBYLA()
    qaoa = QAOA(optimizer, reps=n_layers, quantum_instance=Aer.get_backend('statevector_simulator'))

    result = qaoa.compute_minimum_eigenvalue(qubit_op)
    print(result)

    most_likely_state = max_cut.sample_most_likely(result.eigenstate)
    print("Measurement result with Qiskit:\n", most_likely_state)
    print(f'The max number of crossing edges computed by QAOA is {qp.objective.evaluate(most_likely_state)}')
