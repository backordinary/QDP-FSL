# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/LiH_sympreserving_experiments.py
import numpy as np
from qiskit import *

#print(qiskit.__qiskit_version__)

from VQE import *
from sym_preserving_state_ansatze import *

def print_lih_hamiltonian():

    dist = 1.6 # Ångstrøm

    lih_geometry = ["Li 0.0 0.0 0.0", "H 0.0 0.0 1.6"]

    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 0.74"]

    qubit_op_lih = get_qubit_operator(lih_geometry)

    qubit_op_h2 = get_qubit_operator(h2_geometry)

    print(qubit_op_lih[0].paulis)
    print(qubit_op_h2)

def lih_sympreserving_vqe_singledistance_statevectorsim():
    dist = 1.6

    experiment_name = "lih_sympreserving_dist{:.2f}_statevectorsim_3".format(dist)
    lih_geometry = ["Li 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    method = "COBYLA"
    tol = 0.01

    ansatz, parameters = get_n12_m2_parametrized_ansatz()

    vqe = VQE(geometry=lih_geometry, ansatz=ansatz, parameters=parameters,
              save_results=True, experiment_name=experiment_name)

    x0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(len(parameters))])
    print("x0:", x0)

    energy, params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)
    print("Energy:", energy)
    print("params:", params)

def lih_sympreserving_vqe_singledistance_statevectorsim_alt():
    dist = 1.6

    experiment_name = "lih_sympreserving_dist{:.2f}_statevectorsim_alt2".format(dist)
    lih_geometry = ["Li 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    method = "COBYLA"
    tol = 0.01

    ansatz, parameters = get_n12_m4_parametrized_ansatz()

    vqe = VQE(geometry=lih_geometry, ansatz=ansatz, parameters=parameters,
              save_results=True, experiment_name=experiment_name)

    x0 = np.array([np.pi + 0.1 for i in range(len(parameters))])
    print("x0:", x0)

    energy, params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)
    print("Energy:", energy)
    print("params:", params)

    experiment_name = "lih_sympreserving_dist{:.2f}_statevectorsim_alt3".format(dist)

    vqe = VQE(geometry=lih_geometry, ansatz=ansatz, parameters=parameters,
              save_results=True, experiment_name=experiment_name)

    energy2, params2 = vqe.compute_statevector_vqe(x0=params, method=method, tol=tol, verbose=True)
    print("Energy:", energy2)
    print("params:", params2)


def lih_sympreserving_vqe_distances_statevectorsim():

    return

if __name__ == "__main__":

    lih_sympreserving_vqe_singledistance_statevectorsim_alt()

