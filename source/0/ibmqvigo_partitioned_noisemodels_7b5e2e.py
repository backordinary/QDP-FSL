# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/ibmqvigo_partitioned_noisemodels.py
from qiskit import *

from qiskit.result.result import Result
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo, FakeLondon

from qiskit.transpiler import PassManager

import numpy as np

import os, sys, pickle

abs_path = os.path.dirname(__file__)
sys.path.append(abs_path)
sys.path.append(os.path.dirname(abs_path))

#from error_mitigation.zero_noise_extrapolation_cnot import ZeroNoiseExtrapolation
from swaptest_circuit import create_swap_circuit, swaptest_exp_val_func, qc_swaptest

"""
CONSTRUCT NOISE MODELS

"""

def split_noise_model(noise_model: NoiseModel) -> (NoiseModel, NoiseModel, NoiseModel):
    noise_dict = noise_model.to_dict()

    cnot_errors = {"errors": []}
    singleq_errors = {"errors": []}
    measurement_errors = {"errors": []}

    for err in noise_dict["errors"]:
        if err["type"] == "qerror":
            if "cx" in err["operations"]:
                cnot_errors["errors"].append(err)
            else:
                singleq_errors["errors"].append(err)
        elif err["type"] == "roerror":
            measurement_errors["errors"].append(err)
        else:
            print(err["type"], "not recognised")
    cnot_and_meas_errors = {"errors": cnot_errors["errors"] + measurement_errors["errors"]}
    cnot_and_singleq_errors = {"errors": cnot_errors["errors"] + singleq_errors["errors"]}

    return NoiseModel.from_dict(cnot_errors), NoiseModel.from_dict(cnot_and_meas_errors), \
           NoiseModel.from_dict(cnot_and_singleq_errors)

if __name__ == "__main__":

    directory = os.path.dirname(__file__) + "/results"

    N_AMP_FACTORS = 7
    SHOTS = 1024*8192

    mock_backend = FakeLondon()
    sim_backend = Aer.set_up_backend("qasm_simulator")

    qc_swaptest = create_swap_circuit([1], [2], probe=0, n_qubits=3)

    qc_swaptest_transpiled = transpile(qc_swaptest, backend=mock_backend, optimization_level=3)

    # Empty pass manager to ensure no further transpiling
    pass_manager = PassManager()

    noise_model_mockbackend = NoiseModel.from_backend(mock_backend)

    cnot_only_noise, cnot_and_meas_noise, cnot_and_singleq_noise = split_noise_model(noise_model_mockbackend)
    #noise_models = [noise_model_fakevigo, cnot_only_noise, cnot_and_meas_noise, cnot_and_singleq_noise]
    #noise_model_names = ["VIGO: Full noise model", "VIGO: CNOT", "VIGO: CNOT and Measurement", "VIGO: CNOT and Single-Qubit"]
    #experiment_names = ["ibmqvigo_full", "ibmqvigo_cnot", "ibmqvigo_cnot_and_meas", "ibmqvigo_cnot_and_singleq"]

    experiments = {"ibmqlondon_full": {}, "ibmqlondon_cnot": {},
                   "ibmqlondon_cnot_and_meas": {}, "ibmqlondon_cnot_and_singleq": {}}

    for i, name in enumerate(experiments.keys()):
        experiments[name]["mitigated_exp_vals"] = np.zeros(N_AMP_FACTORS)
        experiments[name]["noise_amplified_exp_vals"] = None
        experiments[name]["depths"] = None

    experiments["ibmqlondon_full"]["noise_model"] = noise_model_mockbackend
    experiments["ibmqlondon_cnot"]["noise_model"] = cnot_only_noise
    experiments["ibmqlondon_cnot_and_meas"]["noise_model"] = cnot_and_meas_noise
    experiments["ibmqlondon_cnot_and_singleq"]["noise_model"] = cnot_and_singleq_noise

    for experiment_name in experiments.keys():
        print("-----\nEXPERIMENT: {:}".format(experiment_name))
        nm = experiments[experiment_name]["noise_model"]

        for n_amp_factors in range(2, N_AMP_FACTORS+1):
            print("AMP FACTORS: {:}".format(n_amp_factors))
            zne = ZeroNoiseExtrapolation(qc_swaptest_transpiled, swaptest_exp_val_func, backend=sim_backend,
                                         noise_model=nm, n_amp_factors=n_amp_factors, shots=SHOTS,
                                         save_results=True, experiment_name=experiment_name)

            result = zne.mitigate(verbose=True)

            experiments[experiment_name]["mitigated_exp_vals"][n_amp_factors-1] = result

        experiments[experiment_name]["mitigated_exp_vals"][0] = zne.noise_amplified_exp_vals[0]
        experiments[experiment_name]["noise_amplified_exp_vals"] = zne.noise_amplified_exp_vals
        experiments[experiment_name]["depths"] = zne.depths

    filename = "results/ibmqlondon_partitioned_noisemodels_results"

    file = open(filename, "wb")
    pickle.dump(experiments, file)
    file.close()

    print(experiments)
