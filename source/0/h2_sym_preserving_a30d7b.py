# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/H2_sym_preserving.py
from VQE import *

from qiskit import *
from qiskit.providers.aer.noise import NoiseModel, pauli_error, QuantumError

from qiskit.aqua.algorithms import NumPyEigensolver

from qiskit.test.mock import FakeAthens

from sym_preserving_state_ansatze import *

from H2 import construct_pauli_noise_model, construct_depol_noise_model, construct_cnot_depol_noise_model

def h2_sympreserving_vqe_singledistance_errorrates_allcombinations():
    backend = Aer.get_backend("qasm_simulator")

    n_params = 10

    statevec_ansatz, statevec_parameters = get_n4_m2_parameterized_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_n4_m2_parameterized_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_n4_m2_errordetection_circuit()
    error_detect_decision_rule = n4_m2_errordetection_decision_rule

    n_amp_factors = 3
    shots_per = 8192
    repeats = 128
    shots = repeats * shots_per

    error_rates = np.asarray([0.001, 0.005, 0.01, 0.015, 0.02])

    num = len(error_rates)
    energies = {"statevector": np.zeros(num), "depolnoise": np.zeros(num), "depolnoise, error detection": np.zeros(num),
                "depolnoise, zne": np.zeros(num), "depolnoise, zne and error detection": np.zeros(num),
                "exact": np.zeros(num)}
    variances = {"depolnoise": np.zeros((num, 1)), "depolnoise, error detection": np.zeros((num, 1)),
                 "depolnoise, zne": np.zeros((num, n_amp_factors)),
                 "depolnoise, zne and error detection": np.zeros((num, n_amp_factors)),
                }
    discarded = {"error detection": np.zeros((num, 1)), "zne and error detection": np.zeros((num, n_amp_factors))}

    params = np.zeros((num, n_params))

    dist = 0.74

    x0 = np.asarray([np.pi / 2 + 0.5 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.0001

    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    for i, p in enumerate(error_rates):

        print("ERROR RATE:", p)
        noise_model = construct_depol_noise_model(p=p)

        experiment_name_statevec = "h2_sympreserving_dist{:.3f}_statevector".format(dist)
        experiment_name = "h2_sympreserving_dist{:.3f}_depolcnotp{:.3f}".format(dist, p)

        # statevector
        print("statevector vqe")

        vqe = VQE(geometry=h2_geometry, ansatz=statevec_ansatz, parameters=statevec_parameters,
                  backend=backend, shots=shots,
                  save_results=True, experiment_name=experiment_name_statevec,
                  )

        energy, opt_params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)
        #energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies["statevector"][i] = energy
        params[i] = opt_params
        print("E=", energy, "\n____")

        print("exact energy")

        energy = vqe.compute_exact_energy()

        energies["exact"][i] = energy

        print("E=", energy, "\n____")

        # NOISY SIMULATIONS:

        # DEPOLNOISE, ZERO NOISE EXTRAPOLATION
        print("DEPOLNOISE, ZNE")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  n_amp_factors=n_amp_factors,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        noise_amplified_energies = vqe.noise_amplified_energies

        # BARE CIRCUIT NOISY RESULTS
        energies["depolnoise"][i] = np.asarray([noise_amplified_energies[0]])
        variances["depolnoise"][i] = np.asarray([vqe.variances[0]])

        # ZERO NOISE EXTRAPOLATION RESULTS
        energies["depolnoise, zne"][i] = energy
        variances["depolnoise, zne"][i] = vqe.variances

        print("Bare circuit noisy results:")
        print("E=", energies["depolnoise"][i])
        print("var=", variances["depolnoise"][i])

        print("Zero-noise extrapolation results")
        print("E=", energy)
        print("var=", vqe.variances)
        print("____")



        # DEPOLNOISE, ZERO NOISE EXTRAPOLATION AND ERROR DETECTION
        print("depolnoise, zne and error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=error_detect_decision_rule,
                  n_amp_factors=n_amp_factors,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        noise_amplified_energies = vqe.noise_amplified_energies

        # ZNE and error detection results
        energies["depolnoise, zne and error detection"][i] = energy
        discarded["zne and error detection"][i] = vqe.discarded_rates
        variances["depolnoise, zne and error detection"][i] = vqe.variances

        # Error detection results
        energies["depolnoise, error detection"][i] = np.asarray([noise_amplified_energies[0]])
        discarded["error detection"][i] = np.asarray([vqe.discarded_rates[0]])
        variances["depolnoise, error detection"][i] = np.asarray([vqe.variances[0]])

        print("Error detection results:")
        print("E=", energies["depolnoise, error detection"][i])
        print("discarded_rates=", variances["depolnoise, error detection"][i])
        print("var=", variances["depolnoise, error detection"][i])

        print("ZNE and error detection results:")
        print("E=", energy, "\n____")
        print("discarded_rates=", vqe.discarded_rates)
        print("var=", vqe.variances)
        print("____")

    print("FINAL: energies")
    print(energies)
    print("FINAL: discarded rates")
    print(discarded)

    filename = "h2_sympreserving_vqe_errorrates_depolcnot_{:}repeats.npz".format(str(repeats))

    np.savez("results/" + filename, params=params, x0=x0, method=method, tol=tol,
             error_rates=error_rates, dist=dist,
             energies_statevector=energies["statevector"],
             energies_exact=energies["exact"],
             energies_depolnoise=energies["depolnoise"],
             energies_errordetect=energies["depolnoise, error detection"],
             energies_depolnoise_zne=energies["depolnoise, zne"],
             energies_depolnoise_zne_errordetect=energies["depolnoise, zne and error detection"],

             discarded_errordetect=discarded["error detection"],
             discarded_zne_errordetect=discarded["zne and error detection"],

             variances_depolnoise=variances["depolnoise"],
             variances_errordetect=variances["depolnoise, error detection"],
             variances_depolnoise_zne=variances["depolnoise, zne"],
             variances_depolnoise_zne_errordetect=variances["depolnoise, zne and error detection"],
             )

    print("DONE")

if __name__ == "__main__":
    h2_sympreserving_vqe_singledistance_errorrates_allcombinations()