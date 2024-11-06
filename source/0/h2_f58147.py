# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/H2.py
from VQE import *

from qiskit import *
from qiskit.quantum_info import Kraus
from qiskit.providers.aer.noise import NoiseModel, pauli_error, QuantumError

from qiskit.test.mock import FakeAthens

def construct_pauli_noise_model(p_cnot: float = 0.01, p_single_q: float = 0.001, p_meas: float = 0.05,
                                single_qubit_noise: bool = True, measurement_noise: bool = True):
    noise_model = NoiseModel()

    # CNOT noise
    cnot_bitflip_error = pauli_error([("X", p_cnot), ("I", 1-p_cnot)])
    cnot_phaseflip_error = pauli_error([("Z", p_cnot), ("I", 1-p_cnot)])

    cnot_pauli_error_composed = cnot_bitflip_error.compose(cnot_phaseflip_error)
    cnot_pauli_error = cnot_pauli_error_composed.tensor(cnot_pauli_error_composed)

    noise_model.add_all_qubit_quantum_error(cnot_pauli_error, ["cx"])

    # Single-qubit noise
    if single_qubit_noise:
        single_q_bitflip_error = pauli_error([("X", p_single_q), ("I", 1-p_single_q)])
        single_q_phaseflip_error = pauli_error([("Z", p_single_q), ("I", 1-p_single_q)])

        single_q_pauli_error = single_q_bitflip_error.compose(single_q_phaseflip_error)

        noise_model.add_all_qubit_quantum_error(single_q_pauli_error, ["u2", "u3"])

    # Measurement noise
    if measurement_noise:
        meas_bitflip_error = pauli_error([("X", p_meas), ("I", 1-p_meas)])
        meas_phaseflip_error = pauli_error([("Z", p_meas), ("I", 1-p_meas)])

        meas_pauli_error = meas_bitflip_error.compose(meas_phaseflip_error)

        noise_model.add_all_qubit_quantum_error(meas_pauli_error, ["measure"])

    return noise_model

def construct_depol_noise_model(p: float):
    if p > 1.0:
        raise Exception("Invalid probability, must be p <= 1.0")

    X = np.asarray([[0,1],[1,0]])
    Y = np.asarray([[0, -1j],[1j,0]])
    Z = np.asarray([[1,0],[0,-1]])
    I = np.asarray([[1,0],[0,1]])

    kraus_operators = [np.sqrt(1-p)*I, np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z]

    error = QuantumError(noise_ops=kraus_operators)
    cx_error = error.tensor(error)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(cx_error, ["cx"])

    return noise_model

def construct_cnot_depol_noise_model(p: float = 0.001):
    if p > 1.0:
        raise Exception("Invalid probability, must be p <= 1.0")

    X = np.asarray([[0, 1], [1, 0]])
    Y = np.asarray([[0, -1j], [1j, 0]])
    Z = np.asarray([[1, 0], [0, -1]])
    I = np.asarray([[1, 0], [0, 1]])

    pauli_dict = {"X": X, "Y": Y, "Z": Z, "I": I}

    kraus_operators = [np.sqrt(1-p)*np.kron(I,I)]

    for a in ["I","X","Y","Z"]:
        for b in ["I","X","Y","Z"]:
            if not ((a=="I") and (b=="I")):
                op = np.kron(pauli_dict[a], pauli_dict[b])
                kraus_operators.append(np.sqrt(p/15)*op)

    error = QuantumError(noise_ops=kraus_operators, number_of_qubits=2)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ["cx"])

    return noise_model

from sym_preserving_state_ansatze import *
from uccsd_state_ansatze import *

ANSATZ, PARAMS = get_n4_m2_parameterized_ansatz()
#ANSATZ_STATEVEC, PARAMS_STATEVEC = get_n4_m2_parameterized_ansatz(n_qubits=4, n_cbits=4)

def h2_vqe_distances_statevectorsim():
    min_dist = 0.5
    max_dist = 3.0
    num = 20

    distances = np.linspace(min_dist, max_dist, num)
    results = np.zeros(num)
    params = np.zeros((num, 10))

    backend = Aer.get_backend("qasm_simulator")
    opt_method = "COBYLA"

    for i, dist in enumerate(distances):
        pass

def h2_vqe_distances_noisefree():
    """
    Compute ground state energy of H2 at a range of intermolecular distances
    using a noisefree qasm_simulator.
    """

    min_dist = 0.5
    max_dist = 3.0
    num = 20

    distances = np.linspace(min_dist, max_dist, num)
    results = np.zeros(num)
    params = np.zeros((num, 10))

    backend = Aer.get_backend("qasm_simulator")
    opt_method = "COBYLA"
    shots = 10*8192

    for i, dist in enumerate(distances):
        experiment_name = "h2_dist{:.2f}_noisefree".format(dist)
        h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

        vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
                  backend=backend, shots=shots,
                  save_results=True, experiment_name=experiment_name,
                  )

        x0 = np.asarray([np.pi/2 for i in range(10)])

        result, optimal_params = vqe.compute_vqe(x0, method=opt_method)

        print(result)

        results[i], params[i,:] = results, optimal_params

    print("DONE:", results)

    filename = "h2_vqe_distances_noisefree_mindist{:}_maxdist{:}_num{:}_optmethod{:}.npz".format(min_dist, max_dist,
                                                                                                 num, opt_method)
    np.savez(filename, results=results, params=params, distances=distances, method=opt_method)

def h2_distances_exactenergies():
    """
    Compute the exact ground state energies for H2 at a range of distances.

    :return:
    """
    return

"""
-- SINGLE DISTANCE EXPERIMENT, H2 at a=0.74 Å
"""


def h2_vqe_singledistance_statevectorsim():
    """
    Compute ground state energy and optimal variational parameters for H2 at dist=0.74 using
    a noiseless state vector simulator.

    The optimal parameters will be used for later energy computation using different noise models.

    :return:
    """

    dist = 0.74

    experiment_name = "h2_dist0.74_statevectorsim"
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    method = "COBYLA"
    tol = 0.0001

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ_STATEVEC, parameters=PARAMS_STATEVEC,
              save_results=True, experiment_name=experiment_name,
              )

    x0 = np.asarray([np.pi + 0.5 for i in range(10)])

    energy, params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol)
    print(energy, params)

def h2_energycomputation_at_optimalparams_singledistance():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("____")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: Optimal params, noise free")

    backend = Aer.get_backend("qasm_simulator")
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    experiment_name = "h2_dist{:.2f}_paulinoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
              backend=backend, shots=shots,
              save_results=True, experiment_name=experiment_name,
              )

    energy = vqe.objective_function(params)

    print(energy)

def h2_energycomputation_at_optimalparams_singledistance_paulinoise():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("____")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: Optimal params, pauli noise")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_pauli_noise_model()
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    experiment_name = "h2_dist{:.2f}_paulinoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
              backend=backend, noise_model=noise_model, shots=shots,
              save_results=True, experiment_name=experiment_name,
              )

    energy = vqe.objective_function(params)

    print(energy)

def h2_energycomputation_at_optimalparams_singledistance_paulinoise_errordetection():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("Energy:", energy)
    print("Params:", params)

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_pauli_noise_model()
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    error_detect_qc = get_n4_m2_errordetection_circuit()

    experiment_name = "h2_dist{:.2f}_paulinoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
              backend=backend, noise_model=noise_model, shots=shots,
              error_detect_qc=error_detect_qc, error_detect_qubits=[4],
              save_results=True, experiment_name=experiment_name,
              )

    print(vqe.measurement_circuits)

    energy = vqe.objective_function(params)

    print(energy)

def h2_energycomputation_at_optimalparams_singledistance_paulinoise_zne():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("Energy:", energy)
    print("Params:", params)

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_pauli_noise_model()
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    experiment_name = "h2_dist{:.2f}_paulinoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
              backend=backend, noise_model=noise_model, shots=shots,
              save_results=False, experiment_name=experiment_name,
              n_amp_factors=3
              )

    energy = vqe.objective_function(params)

    print(energy)

def h2_energycomputation_at_optimalparams_singledistance_paulinoise_zne_and_errordetection():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("Energy:", energy)
    print("Params:", params)

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_pauli_noise_model()
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]
    ansatz, parameters = get_n4_m2_parameterized_ansatz(n_qubits=8, n_cbits=8)
    error_detect_qc, error_detect_qubits = get_errordetecqc_n4_m2_particlenum_4ancillas(tot_qubits=8, tot_clbits=8)
    decision_rule = decision_rule_n4_m2_particlenum_4ancilllas

    ansatz.draw(output="mpl")
    plt.show()

    error_detect_qc.draw(output="mpl")
    plt.show()

    experiment_name = "h2_dist{:.2f}_paulinoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, noise_model=noise_model, shots=shots,
              error_detect_qc=error_detect_qc, decision_rule=decision_rule, error_detect_qubits=error_detect_qubits,
              save_results=False, experiment_name=experiment_name,
              n_amp_factors=3
              )

    vqe.ansatz.draw(output="mpl")
    plt.show()

    energy = vqe.objective_function(params)

    print(energy)

"""
--- DEPOL NOISE
"""

def h2_energycomputation_at_optimalparams_singledistance_depolnoise():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("----")
    print("Energy:", energy)
    print("Params:", params)
    print("----\nResults, depol noise")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=0.001)
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    experiment_name = "h2_dist{:.2f}_depolnoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
              backend=backend, noise_model=noise_model, shots=shots,
              save_results=False, experiment_name=experiment_name,
              )

    energy = vqe.objective_function(params)

    print(energy)

def h2_energycomputation_at_optimalparams_singledistance_depolnoise_errordetection():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("Energy:", energy)
    print("Params:", params)
    print("----\nResults, depolnoise, error detection")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=0.01)
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    error_detect_qc = get_n4_m2_errordetection_circuit()

    experiment_name = "h2_dist{:.2f}_depolnoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
              backend=backend, noise_model=noise_model, shots=shots,
              error_detect_qc=error_detect_qc, error_detect_qubits=[4],
              save_results=True, experiment_name=experiment_name,
              )

    print(vqe.measurement_circuits)

    energy = vqe.objective_function(params)

    print(energy)

def h2_energycomputation_at_optimalparams_singledistance_depolnoise_zne():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("Energy:", energy)
    print("Params:", params)
    print("----\nResults, depolnoise, zne")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=0.01)
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]

    experiment_name = "h2_dist{:.2f}_depolnoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ANSATZ, parameters=PARAMS,
              backend=backend, noise_model=noise_model, shots=shots,
              save_results=False, experiment_name=experiment_name,
              n_amp_factors=3
              )

    energy = vqe.objective_function(params)

    print(energy)

def h2_energycomputation_at_optimalparams_singledistance_depolnoise_zne_and_errordetection():
    filename_params = "h2_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("Energy:", energy)
    print("Params:", params)
    print("----\nResults, depolnoise, error detection + ZNE")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=0.01)
    shots = 10 * 8192

    dist = 0.74
    h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])]
    ansatz, parameters = get_n4_m2_parameterized_ansatz(n_qubits=8, n_cbits=8)
    error_detect_qc, error_detect_qubits = get_errordetecqc_n4_m2_particlenum_4ancillas(tot_qubits=8, tot_clbits=8)
    decision_rule = decision_rule_n4_m2_particlenum_4ancilllas

    ansatz.draw(output="mpl")
    plt.show()

    error_detect_qc.draw(output="mpl")
    plt.show()

    experiment_name = "h2_dist{:.2f}_depolnoise".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, noise_model=noise_model, shots=shots,
              error_detect_qc=error_detect_qc, decision_rule=decision_rule, error_detect_qubits=error_detect_qubits,
              save_results=False, experiment_name=experiment_name,
              n_amp_factors=3
              )

    vqe.ansatz.draw(output="mpl")
    plt.show()

    energy = vqe.objective_function(params)

    print(energy)

#

def h2_sympreserving_vqe_singledistance_errorrates_allcombinations():
    backend = Aer.get_backend("qasm_simulator")

    n_params = 10

    statevec_ansatz, statevec_parameters = get_n4_m2_parameterized_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_n4_m2_parameterized_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_n4_m2_errordetection_circuit()
    decision_rule = n4_m2_errordetection_decision_rule

    n_amp_factors = 3
    shots = 10 * 8192
    max_shots = 150 * 8192
    error_tol = 0.001

    error_rates = np.asarray([0.001, 0.005, 0.01, 0.015, 0.02])

    num = len(error_rates)
    energies = {"statevector": np.zeros(num), "depolnoise": np.zeros(num), "depolnoise, error detection": np.zeros(num),
                "depolnoise, zne": np.zeros(num), "depolnoise, zne and error detection": np.zeros(num),
                "exact": np.zeros(num)}
    variances = {"depolnoise": np.zeros((num, 1)), "depolnoise, error detection": np.zeros((num, 1)),
                 "depolnoise, zne": np.zeros((num, n_amp_factors)),
                 "depolnoise, zne and error detection": np.zeros((num, n_amp_factors)),
                }
    performed_shots = {"depolnoise": np.zeros((num, 1)), "depolnoise, error detection": np.zeros((num, 1)),
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

        # statevector
        print("statevector vqe")

        experiment_name = "h2_n4m2sympres_dist{:.3f}_statevector".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=statevec_ansatz, parameters=statevec_parameters,
                  backend=backend, shots=shots,
                  save_results=True, experiment_name=experiment_name,
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

        # Depolnoise, no error mitigation
        print("depolnoise energy calc")

        experiment_name = "h2_n4m2sympres_dist{:.3f}_depol_p{:.3f}".format(dist, p)

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise"][i] = energy
        variances["depolnoise"][i] = vqe.variances
        performed_shots["depolnoise"][i] = vqe.performed_shots

        print("E=", energy, "\n____")

        # Depolnoise, error detection
        print("depolnoise, error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=n4_m2_errordetection_decision_rule,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, error detection"][i] = energy
        discarded["error detection"][i] = vqe.discarded_rates
        variances["depolnoise, error detection"][i] = vqe.variances
        performed_shots["depolnoise, error detection"][i] = vqe.performed_shots

        print("E=", energy, "\n____")

        # Depolnoise, zero noise extrapolation
        print("depolnoise, zne")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  n_amp_factors=3,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, zne"][i] = energy
        variances["depolnoise, zne"][i] = vqe.variances
        performed_shots["depolnoise, zne"][i] = vqe.performed_shots

        print("E=", energy, "\n____")

        # Depolnoise, zero noise extrapolation and error detection
        print("depolnoise, zne and error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=n4_m2_errordetection_decision_rule,
                  n_amp_factors=3,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, zne and error detection"][i] = energy
        discarded["zne and error detection"][i] = vqe.discarded_rates
        variances["depolnoise, zne and error detection"][i] = vqe.variances
        performed_shots["depolnoise, zne and error detection"][i] = vqe.performed_shots

        print("E=", energy, "\n____")

    print(energies)

    filename = "h2_n4m2sympres_vqe_errorrates_depolnoisemodel.npz"

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

             performed_shots_depolnoise=performed_shots["depolnoise"],
             performed_shots_errordetect=performed_shots["depolnoise, error detection"],
             performed_shots_depolnoise_zne=performed_shots["depolnoise, zne"],
             performed_shots_depolnoise_zne_errordetect=performed_shots["depolnoise, zne and error detection"],
             )

if __name__ == "__main__":
    #h2_vqe_singledistance_statevectorsim()

    h2_energycomputation_at_optimalparams_singledistance()

    #h2_energycomputation_at_optimalparams_singledistance_paulinoise()

    #h2_energycomputation_at_optimalparams_singledistance_paulinoise_errordetection()

    #h2_energycomputation_at_optimalparams_singledistance_paulinoise_zne()

    #h2_energycomputation_at_optimalparams_singledistance_paulinoise_zne_and_errordetection()


    #h2_energycomputation_at_optimalparams_singledistance_depolnoise()

    #h2_energycomputation_at_optimalparams_singledistance_depolnoise_errordetection()

    #h2_energycomputation_at_optimalparams_singledistance_depolnoise_zne()

    #h2_energycomputation_at_optimalparams_singledistance_depolnoise_zne_and_errordetection()

    #h2_UCCSD_vqe_singledistance_statevectorsim()

