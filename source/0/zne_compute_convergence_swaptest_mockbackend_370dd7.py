# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/computation_files/zne_compute_convergence_swaptest_mockbackend.py
from qiskit import *
from qiskit.test.mock import FakeVigo
import numpy as np

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)
path.append(abs_path + "/computation_files")

from zne_circuits import qc_swaptest, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import *

if __name__ == "__main__":
    FILENAME = abs_path + "/data_files" + "/zne_convergence_mockbackend.npz"

    FILENAME_OBJ = abs_path + "/data_files" + "/zne_convergence_mockbackend_obj.npz"

    file_obj = np.load(FILENAME_OBJ, allow_pickle=True)
    qem = file_obj["qem"][()]

    file_data = np.load(FILENAME, allow_pickle=True)
    repeats_tot = file_data["repeats"]
    all_exp_vals, bare_exp_vals, mitigated_exp_vals \
        = file_data["all_exp_vals"], file_data["bare_exp_vals"], file_data["mitigated_exp_vals"]
    counts = file_data["counts"]

    print(repeats_tot)

    repeats = 500
    qem.mitigate(repeats=repeats, verbose=True)

    #all_exp_vals = np.concatenate((all_exp_vals, qem.all_exp_vals))
    #bare_exp_vals = np.concatenate((bare_exp_vals, qem.bare_exp_vals))
    #mitigated_exp_vals = np.concatenate((mitigated_exp_vals, qem.mitigated_exp_vals))

    #counts = np.concatenate((counts, qem.counts))
    #results = list(results).append(qem.result)

    repeats_tot += repeats
    print(repeats_tot)

    #np.savez(FILENAME, backend_name=mock_backend.name(), n_noise_amplification_factors=n_noise_amplification_factors,
    #         repeats=repeats, shots_per_repeat=shots_per_repeat, all_exp_vals=qem.all_exp_vals,
    #         bare_exp_vals=qem.bare_exp_vals, mitigated_exp_vals=qem.mitigated_exp_vals,
    #         depths=qem.depths, results=qem.result, pauli_twirl=qem.pauli_twirl, counts=qem.counts)

    np.savez(FILENAME, backend_name=file_data["backend_name"], shots_per_repeat=file_data["shots_per_repeat"],
             repeats=repeats_tot, n_noise_amplification_factors=file_data["n_noise_amplification_factors"],
             depths=file_data["depths"], pauli_twirl=file_data["pauli_twirl"],
             all_exp_vals=np.concatenate((all_exp_vals, qem.all_exp_vals)),
             bare_exp_vals=np.concatenate((bare_exp_vals, qem.bare_exp_vals)),
             mitigated_exp_vals=np.concatenate((mitigated_exp_vals, qem.mitigated_exp_vals)),
             counts=np.concatenate((counts, qem.counts)))
