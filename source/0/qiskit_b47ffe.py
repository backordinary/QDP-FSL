# https://github.com/NourAbdelmoneim/NYUADHackathon2022_Team15/blob/cb084d6a2a6caeafe75b75c4371765667fdee797/qiskit/qiskit.py
#!/usr/bin/env python
# coding: utf-8

#import matplotlib.pyplot as plt
#import matplotlib.axes as axes
import numpy as np
import networkx as nx

import qiskit
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from itertools import permutations

def get_path(filename):
    adj_matrix1 = np.loadtxt(filename, delimiter = ' ')
     
    G = nx.from_numpy_matrix(adj_matrix1)
    tsp = Tsp(G)

    adj_matrix = nx.to_numpy_matrix(tsp.graph)
    print("distance\n", adj_matrix)

    qp = tsp.to_quadratic_program()
    print(qp.export_as_lp_string())

    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    qubitOp, offset = qubo.to_ising()
    print("Offset:", offset)
    print("Ising Hamiltonian:")
    print(str(qubitOp))

    algorithm_globals.random_seed = 123
    seed = 10598
    backend = Aer.get_backend("aer_simulator_statevector")
    quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    algorithm_globals.massive=True
    spsa = SPSA(maxiter=300)
    ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
    vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)

    result = vqe.compute_minimum_eigenvalue(qubitOp)

    print("energy:", result.eigenvalue.real)
    print("time:", result.optimizer_time)
    x = tsp.sample_most_likely(result.eigenstate)
    print("feasible:", qubo.is_feasible(x))
    z = tsp.interpret(x)
    print("solution:", z)
    print("solution objective:", tsp.tsp_value(z, adj_matrix))

    solution = ''
    for i in z:
        solution+=str(i)
        solution+=' '
    
    solutionFileName = 'output.txt'

    with open(solutionFileName, 'w') as solutionFile:
        solutionFile.write(solution)

    return solutionFileName


if __name__ == '__main__':
    filename = 'input.txt'
    get_path(filename)