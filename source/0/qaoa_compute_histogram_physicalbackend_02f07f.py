# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/computation_files/qaoa_compute_histogram_physicalbackend.py
from qiskit import *
from qiskit.test.mock import FakeVigo
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from QAOA import QAOA

if __name__ == "__main__":
    GRAPH = nx.Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 3], [2, 3], [0, 3], [3, 4]])

    FILENAME = abs_path + "/data_files" + "/qaoa_histogram_physicalbackend.npz"

    IBMQ_TOKEN = ""     # Copy and paste IBMQ token to this string

    provider = IBMQ.enable_account(IBMQ_TOKEN)

    phys_backend = provider.get_backend("ibmq_valencia")

    #phys_backend = FakeVigo() # mock backend

    shots = 8192

    qaoa = QAOA(graph=GRAPH, backend=phys_backend, shots=shots, xatol=1e-2, fatol=1e-1)

    gamma_0, beta_0 = [1.3], [2.6]

    res = qaoa.run_optimization(gamma_0,beta_0)
    print(res)

    optimized_exp_val = qaoa.optimized_exp_val
    optimized_gamma, optimized_beta = qaoa.optimized_gamma, qaoa.optimized_beta

    result = qaoa.execute_circuit(gamma=optimized_gamma, beta=optimized_beta)

    counts = result.results[0].data.counts

    np.savez(FILENAME, counts=counts, optimized_exp_val=optimized_exp_val, initial_gamma=gamma_0,
             initial_beta=beta_0, optimized_gamma=optimized_gamma, optimized_beta=optimized_beta, shots=shots,
             iterations=res.nit, function_evaluations=res.nfev)