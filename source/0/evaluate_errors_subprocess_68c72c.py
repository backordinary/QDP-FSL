# https://github.com/wiktor145/QTrator/blob/de8f4ce389bb9b6168096ab615e5b3b8de2b1c11/evaluate_errors_subprocess.py
import datetime
import pickle
import sys

from qiskit import execute, Aer
from qiskit.providers.aer import QasmSimulator, noise, AerSimulator
from qiskit.quantum_info import state_fidelity
from qiskit.test.mock import FakeMontreal

from convert_qasm_to_qiskit import get_circuit_from_qasm_file

SEPARATOR = ";"

comp_ideal = {}
comp_noise = {}
comp_noise_noise = {}

fidelity_ideal = {}
fidelity_noise = {}
fidelity_noise_noise = {}

results_ideal = {}
results_noise = {}

state_vectors = {}
state_vectors_noise = {}


def results_similarity(r_ideal, r):
    ideal_size = len(r_ideal)
    score = 0.0

    for a in r_ideal:
        score += (abs(r.get(a, 0) - r_ideal[a]) / r_ideal[a]) / ideal_size

    return score


def log(message):
    print("{} - {}".format(datetime.datetime.now(), message))


circuit = sys.argv[1]

method = sys.argv[2]

try:
    backend_hardware = FakeMontreal()

    noise_model = noise.NoiseModel.from_backend(backend_hardware)

    shots = 1024

    backend_qasm = QasmSimulator()
    backend_statevector = Aer.get_backend('statevector_simulator')

    # backend_qasm.set_options(precision='single')
    backend_qasm.set_options(device='GPU')

    backend_statevector.set_options(precision='single')
    backend_statevector.set_options(device='GPU')

    log(str(circuit) + " - " + str(method))
    c = get_circuit_from_qasm_file("result_files/{}_{}.qasm".format(method, circuit))

    # log("Job noise")
    # job_noise = execute(c, backend_qasm, shots=shots, noise_model=noise_model)
    # count_noise = job_noise.result().get_counts()
    #
    # results_noise[(method, circuit)] = count_noise
    #
    # del job_noise
    # del count_noise
    #
    # log("Job ideal")
    # job_ideal = execute(c, backend_qasm, shots=shots)
    # count_ideal = job_ideal.result().get_counts()
    #
    # results_ideal[(method, circuit)] = count_ideal
    #
    # del job_ideal
    # del count_ideal

    c = get_circuit_from_qasm_file("result_files/{}_{}.qasm".format(method, circuit))

    c.remove_final_measurements()

    log("Statevector noise")
    job = backend_statevector.run(c, shots=1, noise_model=noise_model)
    result = job.result()
    sv1 = result.get_statevector()
    state_vectors_noise[(method, circuit)] = sv1

    del job
    del result
    del sv1

    log("Statevector ideal")

    job = backend_statevector.run(c, shots=1)
    result = job.result()
    sv2 = result.get_statevector()
    state_vectors[(method, circuit)] = sv2

    del job
    del result
    del sv2
    del c

    with open('baseline_dict.pickle', 'rb') as handle:
        baseline_dict = pickle.load(handle)

    # result_ideal = baseline_dict["results"][circuit]["ideal"]
    # results_curr_ideal = results_ideal[(method, circuit)]
    #
    # results_curr = results_noise[(method, circuit)]
    #
    # result_noise = baseline_dict["results"][circuit]["noise"]
    #
    # comp_ideal[(method, circuit)] = results_similarity(result_ideal, results_curr_ideal)
    # comp_noise[(method, circuit)] = results_similarity(result_ideal, results_curr)
    # comp_noise_noise[(method, circuit)] = results_similarity(result_noise, results_curr)

    result_ideal = baseline_dict["statevectors"][circuit]["ideal"]
    results_curr_ideal = state_vectors[(method, circuit)]

    results_curr = state_vectors_noise[(method, circuit)]

    result_noise = baseline_dict["statevectors"][circuit]["noise"]

    fidelity_ideal[(method, circuit)] = state_fidelity(result_ideal, results_curr_ideal)
    fidelity_noise[(method, circuit)] = state_fidelity(result_ideal, results_curr)
    fidelity_noise_noise[(method, circuit)] = state_fidelity(result_noise, results_curr)

    print("$$$$$$$$$$$$")
    print()
    # print("My metric ideal ideal -", comp_ideal[(method, circuit)])
    # print("My metric ideal noise -", comp_noise[(method, circuit)])
    # print("My metric noise noise -", comp_noise_noise[(method, circuit)])
    print("Fidelity ideal ideal -", fidelity_ideal[(method, circuit)])
    print("Fidelity ideal noise -", fidelity_noise[(method, circuit)])
    print("Fidelity noise noise -", fidelity_noise_noise[(method, circuit)])
    print()
    print("$$$$$$$$$$$$")

    with open("benchmark_errors_results.csv", "a") as f:
        f.write(method + SEPARATOR + circuit + SEPARATOR)
        # f.write(str(comp_ideal[(method, circuit)]) + SEPARATOR)
        # f.write(str(comp_noise[(method, circuit)]) + SEPARATOR)
        # f.write(str(comp_noise_noise[(method, circuit)]) + SEPARATOR)
        f.write(str(fidelity_ideal[(method, circuit)]) + SEPARATOR)
        f.write(str(fidelity_noise[(method, circuit)]) + SEPARATOR)
        f.write(str(fidelity_noise_noise[(method, circuit)]))
        f.write("\n")

except Exception as e:
    log(f'Error during evaluating errors for {circuit} {method}')
    log(e)
