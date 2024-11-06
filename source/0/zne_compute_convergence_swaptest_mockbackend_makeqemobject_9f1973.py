# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/computation_files/zne_compute_convergence_swaptest_mockbackend_makeqemobject.py
from qiskit import *
from qiskit.test.mock import FakeVigo
import numpy as np

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from zne_circuits import qc_swaptest, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import *


if __name__ == "__main__":
    FILENAME_OBJ = abs_path + "/data_files" + "/zne_convergence_mockbackend_obj.npz"
    FILENAME_DATA = abs_path + "/data_files" + "/zne_convergence_mockbackend.npz"

    mock_backend = FakeVigo()
    n_noise_amplification_factors = 20

    shots_per_repeat = 8192
    repeats = 500

    qem = ZeroNoiseExtrapolation(qc_swaptest, exp_val_func=swaptest_exp_val_func, backend=mock_backend,
                                 n_amp_factors=n_noise_amplification_factors, shots=shots_per_repeat)

    qem.mitigate(repeats=repeats)

    np.savez(FILENAME_OBJ, qem=qem)

    np.savez(FILENAME_DATA, backend_name=mock_backend.name(), n_noise_amplification_factors=n_noise_amplification_factors,
             repeats=repeats, shots_per_repeat=shots_per_repeat, depths=qem.depths, pauli_twirl=qem.pauli_twirl,
             counts=qem.counts, all_exp_vals=qem.all_exp_vals, mitigated_exp_vals=qem.mitigated_exp_vals,
             bare_exp_vals=qem.bare_exp_vals)
