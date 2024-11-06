# https://github.com/danielbultrini/Clean_Dirty_Qubits/blob/e2ff9b8d35b439a3b4128283fc0e750a90b6f2f2/reproduce_figure_5_data.py
from qiskit import Aer
from qiskit.quantum_info import Pauli

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError, depolarizing_error

import numpy as np
import pandas as pd

import file_utils as fu
import hamiltonian_utils as hu
import gradient_utils as gu


def figure_5_data(
    max_layers=100,
    layer_step_size=10,
    num_qubits=4,
    sim_seeds=[0, 1, 2, 3],
    single_qubit_depol_prob=2.425 * 1e-3,
    two_qubit_depol_prob=None,
    calc_grads_built_in=False,
):
    """
        Reproducing similar data for Fig. 5 from https://arxiv.org/pdf/2205.13454.pdf
        This is meant more as a sketch as to how you might write code to reproduce the entire paper,
        but we used proprietary LANL code for the publication which we cannot share. All data is available,
        but the interested reader can reproduce the depolarizing noise model results here and may
        modify this code as desired.
        https://nonhermitian.org/posts/2021/2021-10-07-vqe_program.html


    Args:
        max_layers (int, optional): maximum HVA layers to compute. Defaults to 100.
        layer_step_size (int, optional): Increments to layers per step. Defaults to 10.
        num_qubits (int, optional): Number of qubits for simulation. Defaults to 4.
        sim_seeds (list, optional): List of seeds for simulation. Defaults to [0,1,2,3].
        single_qubit_depol_prob (_type_, optional): Depolarizing noise probability for single qubit gate. Defaults to 2.425*1e-3.
        two_qubit_depol_prob (_type_, optional): Depolarazing noise probability for two qubit gate. Defaults to 2.425*1e-2.
        calc_grads_built_in (bool, optional): Computes gradients with qiskit internal code instead of bespoke, but very slow. Defaults to False.
    Output:
        pandas dataframe of the results
    """
    # define dictionary to store experimental data
    exp_params = {
        "Layers": [],
        "qubits": [],
        r"$n_d$": [],
        "Error rate": [],
        "GradNum": [],
        "Grad": [],
        "SimSeed": [],
        "RunID": [],
        "Noise type": [],
        "Setup:": [],
    }

    # Experiment setup
    if two_qubit_depol_prob is None:
        two_qubit_depol_prob = single_qubit_depol_prob
    for sim_seed in sim_seeds:
        print("Sim seed: ", sim_seed)
        np.random.seed(sim_seed)
        layers = range(1, max_layers + 2, layer_step_size)
        H = hu.tfim_1d_H(num_qubits, 1)

        # some metadata
        noise_type = "depolarising"
        name = "Clean and Dirty"

        # build noise model

        depol_error = depolarizing_error(single_qubit_depol_prob, 1, True)

        ### Clean and dirty setup
        for dirty_qubits in range(
            num_qubits, -1, -1
        ):  # the last dirty_qubits qubits will be dirty, rest clean
            print("     dirty qubits", dirty_qubits)

            noise_model = NoiseModel()

            if dirty_qubits > 0:
                for nd1 in range(dirty_qubits):
                    nd1 = num_qubits-nd1-1 #from the bottom up
                    noise_model.add_quantum_error(
                        depol_error, ["rx", "ry", "rz","I","i","id"], [nd1] # added all different identities just to make sure they are caught.
                    )

            #### Noise model works by having identity operations (rx(0)) after the XX gates that are noisy when a qubit is marked as noisy above.
            #### This allows for the single qubit depolarizing error to be applied after the XX gates without requiring complex custom gate definitions
            #### at the cost of an extra layer of gates, given no optimization. 
            noisy_backend = AerSimulator(
                method="density_matrix", noise_model=noise_model
            )

            # Initialise data structure for results
            gradients_for_each_layer = []

            for n_layers in layers:
                print(
                    "             Computing gradients for layers: {}".format(n_layers)
                )
                np.random.seed(sim_seed + n_layers)
                initial_param_values = np.random.uniform(
                    0, 2 * np.pi, size=2 * n_layers
                )
                prefactor = 1
                shift = np.pi / 2 
                gradients = gu.parameter_shift_gradients_hva(
                    noisy_backend,
                    num_qubits,
                    n_layers,
                    initial_param_values,
                    shift,
                    prefactor,
                )
                gradients_for_each_layer.append(gradients)

                for index, g in enumerate(gradients):
                    exp_params["Layers"].append(n_layers)
                    exp_params["qubits"].append(num_qubits)
                    exp_params[r"$n_d$"].append(dirty_qubits)
                    exp_params["Error rate"].append(single_qubit_depol_prob)
                    exp_params["GradNum"].append(index)
                    exp_params["Grad"].append(g)
                    exp_params["SimSeed"].append(sim_seed)
                    exp_params["RunID"].append(
                        str(num_qubits)
                        + str(list(range(dirty_qubits)))
                        + str(single_qubit_depol_prob)
                        + noise_type
                    )
                    exp_params["Noise type"].append(noise_type)
                    exp_params["Setup:"].append(name)

                if calc_grads_built_in:
                    built_in_grads = gu.qiskit_builtin_gradients(
                        H, num_qubits, n_layers, initial_param_values
                    )
                    print(
                        "         Noise free gradients w Qiskit built in:",
                        built_in_grads,
                    )
                    print(
                        "         Absolute noise free gradients w Qiskit built in:",
                        [np.divide(np.abs(g), num_qubits) for g in built_in_grads],
                    )

        ##### Variable error setup, can be uncommented for results
        # name = "Variable error"
        # for dirty_qubits in range(
        #     num_qubits - 1, 0, -1
        # ):  # the first dirty_qubits qubits will be dirty, rest clean
        #     print("     variable error noisy qubit euivalent", dirty_qubits)

        #     clean_qubits = num_qubits - dirty_qubits
        #     noise_model = NoiseModel()
        #     var_error_one = single_qubit_depol_prob * dirty_qubits / num_qubits
        #     var_error_two = two_qubit_depol_prob * dirty_qubits / num_qubits
        #     depol_error = depolarizing_error(var_error_one, 1)
        #     cx_depol_error = depolarizing_error(var_error_two, 2)

        #     lopsided_depol_pauli_strings = [Pauli(s) for s in ["II", "XI", "ZI", "YI"]]
        #     depol_probs = [
        #         1 - var_error_two,
        #         var_error_two / 3,
        #         var_error_two / 3,
        #         var_error_two / 3,
        #     ]
        #     lopsided_depol_error = QuantumError(
        #         zip(lopsided_depol_pauli_strings, depol_probs)
        #     )

        #     double_depol_pauli_strings = [
        #         Pauli(s) for s in ["II", "XI", "ZI", "YI", "IX", "IZ", "IY"]
        #     ]
        #     double_probs = [
        #         1 - 2 * var_error_two,
        #         var_error_two / 3,
        #         var_error_two / 3,
        #         var_error_two / 3,
        #         var_error_two / 3,
        #         var_error_two / 3,
        #         var_error_two / 3,
        #     ]
        #     double_depol_error = QuantumError(
        #         zip(double_depol_pauli_strings, double_probs)
        #     )

        #     for nd1 in range(num_qubits):
        #         noise_model.add_quantum_error(depol_error, ["rx", "ry", "rz"], [nd1])
        #         for nd2 in range(nd1 + 1, num_qubits):
        #             noise_model.add_quantum_error(
        #                 double_depol_error, ["rxx"], [nd1, nd2]
        #             )
        #             noise_model.add_quantum_error(
        #                 double_depol_error, ["rxx"], [nd2, nd1]
        #             )

        #     noisy_backend = AerSimulator(
        #         method="density_matrix", noise_model=noise_model
        #     )

        #     # Script specific setup
        #     # only set true for small systems and low layers, it is very slow
        #     # Initialise data structure for results
        #     gradients_for_each_layer = []

        #     for n_layers in layers:
        #         print(
        #             "             Computing gradients for layers: {}".format(n_layers)
        #         )
        #         np.random.seed(sim_seed + n_layers)
        #         initial_param_values = np.random.uniform(
        #             0, 2 * np.pi, size=2 * n_layers
        #         )
        #         print(initial_param_values)
        #         prefactor = 1 / 2
        #         shift = np.pi / (4 * prefactor)
        #         gradients = gu.parameter_shift_gradients_hva(
        #             noisy_backend,
        #             num_qubits,
        #             n_layers,
        #             initial_param_values,
        #             shift,
        #             prefactor,
        #         )
        #         gradients_for_each_layer.append(gradients)

        #         for index, g in enumerate(gradients):
        #             exp_params["Layers"].append(n_layers)
        #             exp_params["qubits"].append(num_qubits)
        #             exp_params[r"$n_d$"].append(dirty_qubits)
        #             exp_params["Error rate"].append(single_qubit_depol_prob)
        #             exp_params["GradNum"].append(index)
        #             exp_params["Grad"].append(g)
        #             exp_params["SimSeed"].append(sim_seed)
        #             exp_params["RunID"].append(
        #                 str(num_qubits)
        #                 + str(list(range(dirty_qubits)))
        #                 + str(single_qubit_depol_prob)
        #                 + noise_type
        #             )
        #             exp_params["Noise type"].append(noise_type)
        #             exp_params["Setup:"].append(name)

        #         if calc_grads_built_in:
        #             built_in_grads = gu.qiskit_builtin_gradients(
        #                 H, num_qubits, n_layers, initial_param_values
        #             )
        #             print(
        #                 "         Noise free gradients w Qiskit built in:",
        #                 built_in_grads,
        #             )
        #             print(
        #                 "         Absolute noise free gradients w Qiskit built in:",
        #                 [np.divide(np.abs(g), num_qubits) for g in built_in_grads],
        #             )

    df = pd.DataFrame(exp_params)
    df["Absolute gradient"] = np.abs(df["Grad"]) / df.qubits
    df.to_pickle(fu.exp_filename(exp_params) + ".pkl")
    return df
