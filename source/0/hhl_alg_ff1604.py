# https://github.com/mariuskriukovas/Bakalaurinis/blob/f877821cb75d534b28c1d3d03ae2ac0df5cc68b6/experiments/hhl_alg.py
import numpy as np
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import HHL, NumPyLSsolver
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.reciprocals import LookupRotation

import git.Bakalaurinis.experiments.hhl_functions as hhl_fun


def create_HHL_algorithm(matrix, vector, auxiliary = 3):
    orig_size = len(vector)
    matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

    eigs = hhl_fun.create_eigs(matrix, auxiliary, 50, False)
    num_q, num_a = eigs.get_register_sizes()

    init_state = Custom(num_q, state_vector=vector)

    reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)
    return HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
               init_state, reci, num_q, num_a, orig_size)

def create_HHL_circuit(matrix, vector):
    algorithm = create_HHL_algorithm(matrix, vector, auxiliary=2)
    qc = algorithm.construct_circuit()
    qc.measure_all()
    return qc


def get_statevector_based_result(matrix, vector):
    algorithm = create_HHL_algorithm(matrix, vector, auxiliary=3)
    result = algorithm.run(QuantumInstance(Aer.get_backend('statevector_simulator')))
    print("Solution:\t\t", np.round(result['solution'], 5))
    # print("Probability:\t\t %f" % result['probability_result'])
    # hhl_fun.fidelity(result['solution'], result_ref['solution'])

def get_classical_result(matrix, vector):
    result_ref = NumPyLSsolver(matrix, vector).run()
    print("Classical Solution:\t", np.round(result_ref['solution'], 5))


