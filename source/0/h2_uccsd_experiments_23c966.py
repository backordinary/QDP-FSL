# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/H2_UCCSD_experiments.py
from VQE import *

from qiskit import *
from qiskit.providers.aer.noise import NoiseModel, pauli_error, QuantumError

from qiskit.aqua.algorithms import NumPyEigensolver

from qiskit.test.mock import FakeAthens

from uccsd_state_ansatze import *

from noise_models import construct_pauli_noise_model, construct_depol_noise_model, construct_cnot_depol_noise_model

def h2_UCCSD_vqe_singledistance_statevectorsim():
    dist = 0.74

    experiment_name = "h2_UCCSD_dist0.74_statevectorsim"
    #h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    method = "COBYLA"
    tol = 0.0001

    ansatz, parameters = get_h2_uccsd_ansatz()

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              save_results=True, experiment_name=experiment_name,
              )

    x0 = np.asarray([np.pi + 0.5 for i in range(len(parameters))])

    energy, params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol)
    print(energy, params)

def h2_UCCSD_singledistance_at_optimalparams():
    filename_params = "h2_UCCSD_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("____")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: UCCSD ansatz")

    backend = Aer.get_backend("qasm_simulator")
    shots = 10 * 8192

    dist = 0.74
    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    ansatz, parameters = get_h2_uccsd_ansatz()

    experiment_name = "h2_UCCSD_dist{:.2f}_noiseless".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, shots=shots,
              save_results=False, experiment_name=experiment_name,
              )

    #x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])

    energy = vqe.objective_function(params=params)

    print(energy)

def h2_UCCSD_singledistance_at_optimalparams_depolnoise(p_depol=0.001):
    filename_params = "h2_UCCSD_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("____")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: UCCSD ansatz, depolnoise")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=p_depol)
    shots = 10 * 8192

    dist = 0.74
    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    ansatz, parameters = get_h2_uccsd_ansatz()

    experiment_name = "h2_UCCSD_dist{:.2f}_noiseless".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, shots=shots, noise_model=noise_model,
              save_results=False, experiment_name=experiment_name,
              )

    #x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])

    energy = vqe.objective_function(params=params)

    print(energy)

def h2_UCCSD_singledistance_at_optimalparams_depolnoise_errordetect(p_depol=0.001):
    filename_params = "h2_UCCSD_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("____")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: UCCSD ansatz, depolnoise, errordetect")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=p_depol)
    shots = 8192

    dist = 0.74
    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    experiment_name = "h2_UCCSD_dist{:.3f}_depolnoise_p{:}".format(dist, p_depol)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, shots=shots, noise_model=noise_model,
              save_results=False, experiment_name=experiment_name,
              error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
              decision_rule=decision_rule_uccsd_error_detect, error_controlled_sampling=True, error_tol=0.001,
              )

    #x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])

    energy = vqe.objective_function(params=params, verbose=True)

    print(energy)

def h2_UCCSD_singledistance_at_optimalparams_depolnoise_zne(p_depol=0.001):
    filename_params = "h2_UCCSD_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("____")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: UCCSD ansatz, depolnoise, errordetect")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=p_depol)
    shots = 8192

    dist = 0.74
    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    #error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    experiment_name = "h2_UCCSD_dist{:.3f}_depolnoise_p{:}".format(dist, p_depol)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, shots=shots, noise_model=noise_model,
              save_results=True, experiment_name=experiment_name,
              error_controlled_sampling=True, error_tol=0.001,
              n_amp_factors=3
              )

    # x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])

    energy = vqe.compute_energy(params=params, save_results=True, verbose=True)

    print(energy)

def h2_UCCSD_singledistance_at_optimalparams_depolnoise_errordetect_and_zne(p_depol=0.001):
    filename_params = "h2_UCCSD_dist0.74_statevectorsim_VQE_backendstatevector_simulator_shots1_optmethodCOBYLA_tol0.0001_errordetectFalse"

    file = open("results" + "/" + filename_params + ".energy", "rb")
    energy = pickle.load(file)
    file.close()

    file = open("results" + "/" + filename_params + ".params", "rb")
    params = pickle.load(file)
    file.close()

    print("____")
    print("From statevector vqe")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: UCCSD ansatz, depolnoise, errordetect and zne")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=p_depol)
    shots = 10 * 8192

    dist = 0.74
    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    experiment_name = "h2_UCCSD_dist{:.2f}_noiseless".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, shots=shots, noise_model=noise_model,
              save_results=False, experiment_name=experiment_name,
              error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
              decision_rule=decision_rule_uccsd_error_detect,
              n_amp_factors=3
              )

    #x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])

    energy = vqe.objective_function(params=params)

    print(energy)

# VQE

def h2_UCCSD_vqe_singledistance_depolnoise(p_depol: float = 0.001):
    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=p_depol)
    shots = 8192

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    # error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])
    method = "SPSA"
    tol = 0.001

    dist = 0.74

    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    experiment_name = "h2_UCCSD_dist{:.3f}_depolcnot".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
              backend=backend, shots=shots, noise_model=noise_model,
              save_results=True, experiment_name=experiment_name,
              )

    energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

    filename = "h2_UCCSD_vqe_singledistance074_depolcnot.npz"

    np.savez("results/" + filename, energy=energy, opt_params=opt_params, method=method, tol=tol, x0=x0,
             noise_model=noise_model, p=p_depol, backend_name=backend.name)

def h2_UCCSD_vqe_distances_depolnoise(p_depol: float = 0.001):
    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_cnot_depol_noise_model(p=p_depol)
    shots = 1024 * 8192

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    #error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    min_dist = 0.3
    max_dist = 2.5
    num = 6
    distances = np.linspace(min_dist, max_dist, num)
    energies = np.zeros(num)
    params = np.zeros((num, len(parameters)))
    print(distances)

    x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.0001

    for i, dist in enumerate(distances):
        print("DIST:", dist)

        # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
        h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnot".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  )

        energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies[i], params[i] = energy, opt_params

    print(energies)

    filename = "h2_UCCSD_vqe_distances_depolcnot.npz"

    np.savez("results/" + filename, distances=distances, energies=energies, params=params, method=method, tol=tol,
             shots=shots, noise_model=noise_model, p=p_depol, x0=x0, backend_name=backend.name)

def h2_UCCSD_vqe_distances_depolnoise_errordetect(p_depol: float = 0.001):
    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_cnot_depol_noise_model(p=p_depol)
    shots = 10 * 8192

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    min_dist = 0.3
    max_dist = 2.2
    num = 5
    distances = np.linspace(min_dist, max_dist, num)
    energies = np.zeros(num)
    params = np.zeros((num, len(parameters)))
    print(distances)

    x0 = np.asarray([ 5.58417873, 13.87167908 , 0.71850122, -6.92137263 ,12.49160843, -8.23380742,
                      0.92113021 , 0.78779347 ,-3.45659788 ,-3.96312198, -0.03533009, -3.26684551])
    method = "COBYLA"
    tol = 0.002

    for i, dist in enumerate(distances):
        print("DIST:", dist)

        # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
        h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnot_errordetect".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits, decision_rule=decision_rule_uccsd_error_detect,
                  )

        energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies[i], params[i] = energy, opt_params

    print(energies)

    filename = "h2_UCCSD_vqe_distances_depolcnot_errordetect.npz"

    np.savez("results/" + filename, distances=distances, energies=energies, params=params, method=method, tol=tol,
             shots=shots, noise_model=noise_model, p=p_depol, x0=x0, backend_name=backend.name)

def h2_UCCSD_vqe_distances_depolnoise_zne(p_depol: float = 0.001):
    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=p_depol)
    shots = 2 * 8192

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    # error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    min_dist = 0.3
    max_dist = 2.2
    num = 5
    distances = np.linspace(min_dist, max_dist, num)
    energies = np.zeros(num)
    params = np.zeros((num, len(parameters)))
    print(distances)

    x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.002

    for i, dist in enumerate(distances):
        print("DIST:", dist)

        # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
        h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

        experiment_name = "h2_UCCSD_dist{:.3f}_depolnoise_zne".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  n_amp_factors=3,
                  )

        energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies[i], params[i] = energy, opt_params

    print(energies)

    filename = "h2_UCCSD_vqe_distances_depolnoise_zne.npz"

    np.savez("results/" + filename, distances=distances, energies=energies, params=params, method=method, tol=tol,
             shots=shots, noise_model=noise_model, p=p_depol, x0=x0, backend_name=backend.name)

def h2_UCCSD_vqe_distances_depolnoise_zne_and_errordetect(p_depol: float = 0.001):
    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_depol_noise_model(p=p_depol)
    shots = 10 * 8192

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    min_dist = 0.3
    max_dist = 2.2
    num = 5
    distances = np.linspace(min_dist, max_dist, num)
    energies = np.zeros(num)
    params = np.zeros((num, len(parameters)))
    print(distances)

    x0 = np.asarray([np.pi / 2 + 0.1 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.002

    for i, dist in enumerate(distances):
        print("DIST:", dist)

        # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
        h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

        experiment_name = "h2_UCCSD_dist{:.3f}_depolnoise_zne_and_errordetect".format(dist)

        vqe = VQE(geometry = h2_geometry, ansatz = ansatz, parameters = parameters,
                    backend = backend, shots = shots, noise_model = noise_model,
                    save_results = True, experiment_name = experiment_name,
                    error_detect_qc = error_detect_qc, error_detect_qubits = error_detect_qubits,
                    decision_rule = decision_rule_uccsd_error_detect,
                    n_amp_factors = 3,
                  )

        energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies[i], params[i] = energy, opt_params

    print(energies)

    filename = "h2_UCCSD_vqe_distances_depolnoise_zne_and_errordetect.npz"

    np.savez("results/" + filename, distances=distances, energies=energies, params=params, method=method, tol=tol,
             shots=shots, noise_model=noise_model, p=p_depol, x0=x0, backend_name=backend.name)

# VQE optimal params

def h2_UCCSD_vqe_singledistance_getcounts(p_depol: float = 0.001):
    pass

def h2_UCCSD_vqe_singledistance_zne_noiseamplificationfactors(p_depol: float = 0.01):

    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_cnot_depol_noise_model(p=p_depol)
    shots = 10 * 8192
    dist = 0.74
    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    statevec_ansatz, statevec_parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    x0 = np.asarray([np.pi / 2 + 0.5 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.0001

    experiment_name = "h2_UCCSD_dist{:.3f}_statevector".format(dist)

    vqe = VQE(geometry=h2_geometry, ansatz=statevec_ansatz, parameters=statevec_parameters,
              backend=backend, shots=shots,
              save_results=True, experiment_name=experiment_name,
              )

    energy, params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)

    print("____")
    print("From statevector vqe")
    print("Energy:", energy)
    print("Params:", params)
    print("____\nRESULTS: UCCSD ansatz, depolnoise, errordetect and zne")

    n_amp_factors = 8
    error_tol = 0.001
    max_shots = 200*8192

    mitigated_energies = np.zeros(n_amp_factors)
    mitigated_energies_errordetect = np.zeros(n_amp_factors)

    for n in range(1,n_amp_factors+1):

        print("AMP FACTOR:", n)

        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnotp{:.4f}".format(dist, p_depol)

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  n_amp_factors=n,
                  )

        energy = vqe.compute_energy(params=params, save_results=True, verbose=True)

        print("RESULT, zne with n_amp_factors={:.4f}:".format(n), energy, "\n-----")

        mitigated_energies[n-1] = energy

        vqe_2 = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=decision_rule_uccsd_error_detect,
                  n_amp_factors=n,
                  )

        energy = vqe_2.compute_energy(params=params, save_results=True, verbose=True)

        print("RESULT, zne and error detection with n_amp_factors={:.4f}:".format(n), energy, "\n-----")

        mitigated_energies_errordetect[n-1] = energy

    variances = vqe.variances
    performed_shots = vqe.performed_shots

    variances_errordetect = vqe_2.variances
    performed_shots_errordetect = vqe_2.performed_shots
    discarded_rates_errordetect = vqe_2.discarded_rates

    filename = "h2_UCCSD_depolnoisecnot_p001_amplificationfactors.npz"

    np.savez("results/" + filename,
             mitigated_energies=mitigated_energies,
             mitigated_energies_errordetect=mitigated_energies_errordetect,
             variances=variances,
             performed_shots=performed_shots,
             variances_errordetect=variances_errordetect,
             performed_shots_errordetect=performed_shots_errordetect,
             discarded_rates_errordetect=discarded_rates_errordetect,
             noise_model=noise_model, p_depol=p_depol, n_amp_factors=n_amp_factors, geometry=h2_geometry,
             )

    print(mitigated_energies)
    print(mitigated_energies_errordetect)

def h2_UCCSD_vqe_distances_optimalparamsstatevector_allcombinations(p_depol: float = 0.001):
    backend = Aer.get_backend("qasm_simulator")
    noise_model = construct_cnot_depol_noise_model(p=p_depol)

    n_params = 12

    statevec_ansatz, statevec_parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    n_amp_factors = 5
    shots = 32 * 8192

    error_controlled_sampling = True
    error_tol = 0.0001
    max_shots = 1024 * 8192

    min_dist = 0.3
    max_dist = 2.5
    num = 6
    distances = np.linspace(min_dist, max_dist, num)
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

    print(distances)

    x0 = np.asarray([np.pi / 2 + 0.5 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.0001

    for i, dist in enumerate(distances):
        print("DIST:", dist)

        # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
        h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

        # Statevector
        print("statevector vqe")

        experiment_name = "h2_UCCSD_dist{:.3f}_statevector".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=statevec_ansatz, parameters=statevec_parameters,
                  backend=backend, shots=shots,
                  save_results=True, experiment_name=experiment_name,
                  )

        energy, opt_params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies["statevector"][i] = energy
        params[i] = opt_params
        print("E=",energy,"\n____")

        print("exact energy")

        energy = vqe.compute_exact_energy()

        energies["exact"][i] = energy

        print("E=", energy, "\n____")

        # Depolnoise, no error mitigation
        print("depolnoise energy calc")

        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnotp{:.3f}".format(dist, p_depol)

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=error_controlled_sampling, error_tol=error_tol, max_shots=max_shots,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise"][i] = energy
        variances["depolnoise"][i] = vqe.variances
        performed_shots["depolnoise"][i] = vqe.performed_shots

        print("E=", energy, "\n____")
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("______")

        # Depolnoise, error detection
        print("depolnoise, error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=error_controlled_sampling, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=decision_rule_uccsd_error_detect,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, error detection"][i] = energy
        discarded["error detection"][i] = vqe.discarded_rates
        variances["depolnoise, error detection"][i] = vqe.variances
        performed_shots["depolnoise, error detection"][i] = vqe.performed_shots

        print("E=", energy, "\n____")
        print("discarded=",vqe.discarded_rates)
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("______")

        # Depolnoise, zero noise extrapolation
        print("depolnoise, zne")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=error_controlled_sampling, error_tol=error_tol, max_shots=max_shots,
                  n_amp_factors=n_amp_factors,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, zne"][i] = energy
        variances["depolnoise, zne"][i] = vqe.variances
        performed_shots["depolnoise, zne"][i] = vqe.performed_shots

        print("E=", energy, "\n____")
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("______")

        # Depolnoise, zero noise extrapolation and error detection
        print("depolnoise, zne and error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=decision_rule_uccsd_error_detect,
                  n_amp_factors=n_amp_factors,
                  )

        print("OPERATOR COUNTS:")
        vqe.qc.count_ops()

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, zne and error detection"][i] = energy
        discarded["zne and error detection"][i] = vqe.discarded_rates
        variances["depolnoise, zne and error detection"][i] = vqe.variances
        performed_shots["depolnoise, zne and error detection"][i] = vqe.performed_shots

        print("E=", energy, "\n____")
        print("discarded=", vqe.discarded_rates)
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("______")

    print(energies)

    filename = "h2_UCCSD_vqe_distances_depolcnot.npz"

    np.savez("results/" + filename, params=params, x0=x0, method=method, tol=tol,
             distances=distances, noise_model=noise_model, p_depol=p_depol,
             energies_statevector = energies["statevector"],
             energies_exact = energies["exact"],
             energies_depolnoise = energies["depolnoise"],
             energies_errordetect = energies["depolnoise, error detection"],
             energies_depolnoise_zne = energies["depolnoise, zne"],
             energies_depolnoise_zne_errordetect = energies["depolnoise, zne and error detection"],

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

def h2_UCCSD_vqe_singledistance_errorrates_allcombinations():
    backend = Aer.get_backend("qasm_simulator")

    n_params = 12

    statevec_ansatz, statevec_parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    n_amp_factors = 3
    shots = 16 * 8192
    max_shots = 128 * 8192
    error_tol = 0.0001

    error_rates = np.asarray([0.001, 0.005, 0.01, 0.015, 0.02])

    num = len(error_rates)
    energies = {"statevector": np.zeros(num), "depolnoise": np.zeros(num),
                "depolnoise, error detection": np.zeros(num),
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

    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    for i, p in enumerate(error_rates):
        print("ERROR RATE:", p)
        noise_model = construct_depol_noise_model(p=p)

        # statevector
        print("statevector vqe")

        experiment_name = "h2_UCCSD_dist{:.3f}_statevector".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=statevec_ansatz, parameters=statevec_parameters,
                  backend=backend, shots=shots,
                  save_results=True, experiment_name=experiment_name,
                  )

        energy, opt_params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)
        # energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies["statevector"][i] = energy
        params[i] = opt_params
        print("E=", energy, "\n____")

        print("exact energy")

        energy = vqe.compute_exact_energy()

        energies["exact"][i] = energy

        print("E=", energy, "\n____")

        # DEPOLNOISE, NO ERROR MITIGATION
        print("depolnoise energy calc")

        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnotp{:.3f}".format(dist, p)

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  )

        print("OPERATOR COUNTS:")
        print(vqe.ansatz.count_ops())

        print("HAMILTONIAN:")
        print(vqe.hamiltonian_dict)

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise"][i] = energy
        variances["depolnoise"][i] = vqe.variances
        performed_shots["depolnoise"][i] = vqe.performed_shots

        print("E=", energy, "\n____")
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("____")

        # DEPOLNOISE, ERROR DETECTION
        print("depolnoise, error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=decision_rule_uccsd_error_detect,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, error detection"][i] = energy
        discarded["error detection"][i] = vqe.discarded_rates
        variances["depolnoise, error detection"][i] = vqe.variances
        performed_shots["depolnoise, error detection"][i] = vqe.performed_shots

        print("E=", energy, "\n____")

        # DEPOLNOISE, ZERO NOISE EXTRAPOLATION
        print("depolnoise, zne")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  n_amp_factors=n_amp_factors,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, zne"][i] = energy
        variances["depolnoise, zne"][i] = vqe.variances
        performed_shots["depolnoise, zne"][i] = vqe.performed_shots

        print("E=", energy, "\n____")
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("____")

        # DEPOLNOISE, ZERO NOISE EXTRAPOLATION AND ERROR DETECTION
        print("depolnoise, zne and error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=decision_rule_uccsd_error_detect,
                  n_amp_factors=n_amp_factors,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies["depolnoise, zne and error detection"][i] = energy
        discarded["zne and error detection"][i] = vqe.discarded_rates
        variances["depolnoise, zne and error detection"][i] = vqe.variances
        performed_shots["depolnoise, zne and error detection"][i] = vqe.performed_shots

        print("E=", energy, "\n____")
        print("discarded_rates=", vqe.discarded_rates)
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("____")

    # print(energies)
    print(discarded)
    # print(performed_shots)

    filename = "h2_UCCSD_vqe_errorrates_depolcnot.npz"

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

    print("DONE")

def h2_UCCSD_vqe_singledistance_errorrates_allcombinations_1024repeats():
    backend = Aer.get_backend("qasm_simulator")

    n_params = 12

    statevec_ansatz, statevec_parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    n_amp_factors = 3
    shots = 1024 * 8192

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

    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    for i, p in enumerate(error_rates):

        print("ERROR RATE:", p)
        noise_model = construct_depol_noise_model(p=p)

        # statevector
        print("statevector vqe")

        experiment_name = "h2_UCCSD_dist{:.3f}_statevector".format(dist)

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

        # NOISY SIMULATIONS:

        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnotp{:.3f}".format(dist, p)

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
                  decision_rule=decision_rule_uccsd_error_detect,
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

    filename = "h2_UCCSD_vqe_errorrates_depolcnot_1024repeats.npz"

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


def h2_UCCSD_singledistance_zne_with_errordetect_discardedrates():
    backend = Aer.get_backend("qasm_simulator")

    n_params = 12

    statevec_ansatz, statevec_parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    n_amp_factors = 3
    shots = 16 * 8192
    max_shots = 128 * 8192
    error_tol = 0.0001

    error_rates = np.asarray([0.001, 0.005, 0.01, 0.015, 0.02])

    num = len(error_rates)
    energies = np.zeros((num, n_amp_factors))
    variances = np.zeros((num, n_amp_factors))
    discarded = np.zeros((num, n_amp_factors))

    energies_statevector = np.zeros(num)
    energies_exact = np.zeros(num)

    params = np.zeros((num, n_params))

    dist = 0.74

    x0 = np.asarray([np.pi / 2 + 0.5 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.0001

    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    for i, p in enumerate(error_rates):
        print("ERROR RATE:", p)
        noise_model = construct_depol_noise_model(p=p)

        # statevector
        print("statevector vqe")

        experiment_name = "h2_UCCSD_dist{:.3f}_statevector".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=statevec_ansatz, parameters=statevec_parameters,
                  backend=backend, shots=shots,
                  save_results=True, experiment_name=experiment_name,
                  )

        energy, opt_params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)
        # energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies_statevector[i] = energy
        params[i] = opt_params
        print("E=", energy, "\n____")

        print("exact energy")

        energy = vqe.compute_exact_energy()

        energies_exact[i] = energy

        print("E=", energy, "\n____")


        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnotp{:.3f}".format(dist, p)

        # DEPOLNOISE, ZERO NOISE EXTRAPOLATION AND ERROR DETECTION
        print("depolnoise, zne and error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=decision_rule_uccsd_error_detect,
                  n_amp_factors=n_amp_factors,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies[i,:] = energy
        discarded[i,:] = vqe.discarded_rates
        variances[i,:] = vqe.variances

        print("E=", energy, "\n____")
        print("discarded_rates=", vqe.discarded_rates)
        print("var=", vqe.variances)
        print("performed_shots=", vqe.performed_shots)
        print("____")

    filename = "h2_UCCSD_vqe_zne_with_errordetect_discardedrates.npz"

    np.savez("results/" + filename, error_rates=error_rates, energies=energies, discarded=discarded, variances=variances,
             energies_statevector=energies_statevector, energies_exact=energies_exact)

    print("DONE")


def h2_UCCSD_singledistance_errordetect_discardedrates():
    backend = Aer.get_backend("qasm_simulator")

    n_params = 12

    statevec_ansatz, statevec_parameters = get_h2_uccsd_ansatz(tot_qubits=4, tot_clbits=4)
    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)
    error_detect_qc, error_detect_qubits = get_h2_uccsd_errordetectcircuit_alt()

    shots = 16 * 8192
    max_shots = 128 * 8192
    error_tol = 0.0001

    error_rates = np.asarray([0.001, 0.005, 0.01, 0.015, 0.02])

    num = len(error_rates)
    energies = np.zeros(num)
    variances = np.zeros(num)
    discarded = np.zeros(num)

    energies_statevector = np.zeros(num)
    energies_exact = np.zeros(num)

    params = np.zeros((num, n_params))

    dist = 0.74

    x0 = np.asarray([np.pi / 2 + 0.5 for i in range(len(parameters))])
    method = "COBYLA"
    tol = 0.0001

    # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
    h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

    for i, p in enumerate(error_rates):
        print("ERROR RATE:", p)
        noise_model = construct_depol_noise_model(p=p)

        # statevector
        print("statevector vqe")

        experiment_name = "h2_UCCSD_dist{:.3f}_statevector".format(dist)

        vqe = VQE(geometry=h2_geometry, ansatz=statevec_ansatz, parameters=statevec_parameters,
                  backend=backend, shots=shots,
                  save_results=True, experiment_name=experiment_name,
                  )

        energy, opt_params = vqe.compute_statevector_vqe(x0=x0, method=method, tol=tol, verbose=True)
        # energy, opt_params = vqe.compute_vqe(x0=x0, method=method, tol=tol, verbose=True)

        energies_statevector[i] = energy
        params[i] = opt_params
        print("E=", energy, "\n____")

        print("exact energy")

        energy = vqe.compute_exact_energy()

        energies_exact[i] = energy

        print("E=", energy, "\n____")



        experiment_name = "h2_UCCSD_dist{:.3f}_depolcnotp{:.3f}".format(dist, p)

        # DEPOLNOISE, ERROR DETECTION
        print("depolnoise, error detection")

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz, parameters=parameters,
                  backend=backend, shots=shots, noise_model=noise_model,
                  save_results=True, experiment_name=experiment_name,
                  error_controlled_sampling=True, error_tol=error_tol, max_shots=max_shots,
                  error_detect_qc=error_detect_qc, error_detect_qubits=error_detect_qubits,
                  decision_rule=decision_rule_uccsd_error_detect,
                  )

        energy = vqe.compute_energy(opt_params, save_results=True, verbose=True)

        energies[i] = energy
        discarded[i] = vqe.discarded_rates
        variances[i] = vqe.variances

        print("E=", energy, "\n____")
        print("Discarded rates=", discarded[i])
        print("_____")

    filename = "h2_UCCSD_vqe_errordetect_discardedrates.npz"

    np.savez("results/" + filename, error_rates=error_rates, energies=energies, discarded=discarded, variances=variances,
             energies_statevector=energies_statevector, energies_exact=energies_exact)

    print("DONE")

def h2_distances_exactenergies():

    ansatz, parameters = get_h2_uccsd_ansatz(tot_qubits=6, tot_clbits=6)

    min_dist = 0.3
    max_dist = 2.5
    num = 150
    distances = np.linspace(min_dist, max_dist, num)

    energies = np.zeros(num)

    for i, dist in enumerate(distances):
        print("DIST", dist)
        # h2_geometry = [("H", [0, 0, 0]), ("H", [0, 0, dist])] # Deprecated
        h2_geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 {:}".format(dist)]

        vqe = VQE(geometry=h2_geometry, ansatz=ansatz.copy(), parameters=parameters)

        qubit_op = vqe.qubit_operator
        nuclear_repulsion_energy = vqe.nuclear_repulsion_energy

        result = NumPyEigensolver(qubit_op).run()

        energies[i] = np.real(result.eigenvalues) + nuclear_repulsion_energy

        print("exact energy:", energies[i])

    filename = "h2_distances_exactenergies.npz"

    np.savez("results/" + filename, energies=energies, distances=distances)

if __name__ == "__main__":
    """
    h2_UCCSD_singledistance_at_optimalparams()

    p_depol = 0.001

    h2_UCCSD_singledistance_at_optimalparams_depolnoise(p_depol)

    h2_UCCSD_singledistance_at_optimalparams_depolnoise_errordetect(p_depol)

    h2_UCCSD_singledistance_at_optimalparams_depolnoise_zne(p_depol)

    h2_UCCSD_singledistance_at_optimalparams_depolnoise_errordetect_and_zne(p_depol)
    """

    #h2_UCCSD_vqe_singledistance_depolnoise()

    #h2_UCCSD_vqe_distances_depolnoise()

    #h2_UCCSD_vqe_distances_optimalparamsstatevector_allcombinations()

    #h2_UCCSD_vqe_singledistance_errorrates_allcombinations()

    h2_UCCSD_vqe_singledistance_errorrates_allcombinations_1024repeats()

    #h2_UCCSD_vqe_singledistance_zne_noiseamplificationfactors()

    #h2_UCCSD_singledistance_errordetect_discardedrates()

    #h2_UCCSD_singledistance_zne_with_errordetect_discardedrates()



