# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/computation_files/zne_compute_swaptest_errorcontrolled_mockbackend.py
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
    FILENAME_DATA = abs_path + "/data_files" + "/zne_swaptest_errorcontrolled_mockbackend.npz"

    FILENAME_STAT = abs_path + "/data_files" + "/zne_swaptest_statistics.npz"

    file = np.load(FILENAME_STAT)
    N_s = file["error_sampled_shots"]

    print(np.shape(N_s))

    mock_backend = FakeVigo()

    max_amp_factors = 10

    mitigated = np.zeros(max_amp_factors)

    mitigated[0] = swaptest_exp_val_func(execute(qc_swaptest, backend=mock_backend, shots=8192,
                                                 optimization_level=3).result().get_counts())

    for n in range(2, max_amp_factors+1):
        print("n =",n)
        qem = ZeroNoiseExtrapolation(qc_swaptest, exp_val_func=swaptest_exp_val_func, backend=mock_backend,
                                     n_amp_factors=n, shots=int(N_s[n-2]))

        qem.mitigate(verbose=True)

        mitigated[n-1] = qem.result

    np.savez(FILENAME_DATA, mitigated=mitigated, shots=N_s)

    print(mitigated)
