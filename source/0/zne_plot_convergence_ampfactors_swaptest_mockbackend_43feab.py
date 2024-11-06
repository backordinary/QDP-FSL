# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/plot_files/zne_plot_convergence_ampfactors_swaptest_mockbackend.py
from qiskit import *
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)
path.append(abs_path + "/data_files")

from zne_swaptest_circuit import qc_swaptest, swaptest_exp_val_func
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

    n_amp_factors_included = 8

    dx = 20
    max_repeats = 2000
    num = max_repeats // dx

    rep = np.linspace(1,max_repeats,num=num, endpoint=True,dtype=int)
    tot_shots = rep * 8192

    amp_factors = np.asarray([1+2*i for i in range(n_amp_factors_included)])

    averaged_exp_vals = np.zeros((n_amp_factors_included, num))
    averaged_exp_vals_squared = np.zeros((n_amp_factors_included, num))

    mitigated_3 = np.zeros(num)
    mitigated_5 = np.zeros(num)
    mitigated_8 = np.zeros(num)

    for x, r in enumerate(rep):
        for i in range(n_amp_factors_included):
            averaged_exp_vals[i,x] = np.average(all_exp_vals[0:r,i])
            averaged_exp_vals_squared[i,x] = np.average(all_exp_vals[0:r,i]**2)
        mitigated_3[x] = richardson_extrapolate(averaged_exp_vals[0:3,x],amp_factors[0:3])
        mitigated_5[x] = richardson_extrapolate(averaged_exp_vals[0:5,x],amp_factors[0:5])
        mitigated_8[x] = richardson_extrapolate(averaged_exp_vals[0:8,x],amp_factors[0:8])

    plt.plot(tot_shots, np.zeros(num) + 0.5, linestyle=(0, (5,5)), label=r"E$^*$, true exp val")

    plt.plot(tot_shots, mitigated_3, linestyle=(0,(1,1)), label="E[1,3,5]")
    plt.plot(tot_shots, mitigated_5, linestyle="dashdot", label="E[1,3,5,7,9]")
    plt.plot(tot_shots, mitigated_8, linestyle="solid", label="E[1,3,5,7,9,11,13,15]")

    plt.plot(tot_shots, averaged_exp_vals[0, :], linestyle=(0,(3,1,1,1)), label="E[1], bare circuit")

    plt.xlabel("shots included in averages")

    plt.legend(bbox_to_anchor=(1.02, 1.), loc=2, borderaxespad=0.)

    plt.tight_layout()

    plt.show()


