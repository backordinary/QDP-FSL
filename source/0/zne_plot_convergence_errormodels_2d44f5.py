# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/plot_files/zne_plot_convergence_errormodels.py
from qiskit import *
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)
path.append(abs_path + "/computation_files")

from zne_circuits import qc_swaptest, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import ZeroNoiseExtrapolation, richardson_extrapolate

if __name__ == "__main__":

    FILENAME = abs_path + "/data_files" + "/zne_errormodels_mockbackend.npz"

    file = np.load(FILENAME, allow_pickle=True)

    n_amp_factors_included = file["amp_factors"]

    mitigated_onlycnot = file["mitigated_onlycnot"]
    mitigated_cnotandmeas = file["mitigated_cnotandmeas"]
    mitigated_cnotandsingleq = file["mitigated_cnotandsingleq"]

    xticks = [i+1 for i in range(n_amp_factors_included)]

    plt.plot(xticks, mitigated_onlycnot, 'o--', label=r"$E[1,\dots,2n-1]$")
    plt.plot(xticks, mitigated_cnotandmeas, 'o--', label=r"$E[1,\dots,2n-1]$")
    plt.plot(xticks, mitigated_cnotandsingleq, 'o--', label=r"$E[1,\dots,2n-1]$")

    plt.xticks(ticks=xticks)
    plt.xlabel(r"$n$, number of amplification factors included", fontsize=12)

    plt.legend(fontsize=12)

    plt.tight_layout()

    plt.show()