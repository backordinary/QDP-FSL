# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/plot_files/zne_plot_convergence_swaptest_errorcontrolled.py
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

    FILENAME = abs_path + "/data_files" + "/zne_swaptest_errorcontrolled_mockbackend.npz"

    file = np.load(FILENAME, allow_pickle=True)

    mitigated = file["mitigated"]
    N_s = file["shots"]

    print(N_s)
    print(mitigated)