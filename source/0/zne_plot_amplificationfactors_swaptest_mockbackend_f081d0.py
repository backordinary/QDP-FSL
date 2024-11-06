# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/plot_files/zne_plot_amplificationfactors_swaptest_mockbackend.py
from qiskit import *
import numpy as np
import matplotlib.pyplot as plt

from qiskit.test.mock import FakeVigo

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)
path.append(abs_path + "/computation_files")

from zne_circuits import qc_swaptest, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import ZeroNoiseExtrapolation, richardson_extrapolate

if __name__ == "__main__":

    FILENAME = abs_path + "/data_files" + "/zne_convergence_mockbackend.npz"

    file = np.load(FILENAME, allow_pickle=True)

    backend_name = file["backend_name"]
    n_noise_amplification_factors = file["n_noise_amplification_factors"]
    counts = file["counts"]
    repeats, shots_per_repeat = file["repeats"], file["shots_per_repeat"]
    all_exp_vals, bare_exp_vals, mitigated_exp_vals = file["all_exp_vals"],\
                                                      file["bare_exp_vals"],\
                                                      file["mitigated_exp_vals"]

    mock_backend = FakeVigo()
    if mock_backend.name() != backend_name:
        print("Wrong backend: ", mock_backend.name, "should be:", backend_name)

    qem = ZeroNoiseExtrapolation(qc_swaptest, exp_val_func=swaptest_exp_val_func, backend=mock_backend,
                                 n_amp_factors=n_noise_amplification_factors, shots=shots_per_repeat)

    repeats_included = 5000
    n_amp_factors_included = 15

    noise_amplification_factors = np.asarray([(1 + 2*i) for i in range(n_noise_amplification_factors)])
    x_ticks = [i+1 for i in range(n_amp_factors_included)]
    all_mitigated_exp_vals = [np.average(bare_exp_vals)]

    for i in range(2, n_amp_factors_included+1):
        all_mitigated_exp_vals.append(richardson_extrapolate(np.average(all_exp_vals[0:repeats_included,0:i],axis=0), noise_amplification_factors[0:i]))

    print(all_mitigated_exp_vals)

    plt.xticks(ticks=x_ticks)
    plt.xlabel(r"$n$, number of amplification factors included", fontsize=10)

    plt.plot(x_ticks, np.zeros(len(x_ticks)) + 0.5, 'g--', label="$E^*$, true exp val")
    plt.plot(x_ticks, all_mitigated_exp_vals,'ro--', label=r"$E[1,\dots,2n-1]$, mitigated exp val")

    plt.legend(fontsize=10)

    plt.savefig(abs_path + "/figures" + "/zne_swaptest_convergence_of_n.pdf")

    plt.show()