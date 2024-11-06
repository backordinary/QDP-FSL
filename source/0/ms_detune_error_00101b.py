# https://github.com/aleksey-uvarov/permutevqe/blob/f9550348c0ed9cbcd27cb2b0cf4e0bbfda0d44ac/ms_detune_error.py
"""The same stuff you can see in sample_permutations, but for a detuning-based noise model.
The reason for such a separation is that a more complicated parametric noise model requires
a transpiler pass."""


from qiskit.circuit import Instruction, QuantumCircuit
from qiskit_aer.noise import LocalNoisePass, kraus_error
from qiskit.circuit.library import RXXGate, TwoLocal
from qiskit_aer.backends import AerSimulator
from qiskit.opflow import CircuitSampler, PauliExpectation, PauliSumOp
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.utils import QuantumInstance
from qiskit.opflow import StateFn, CircuitStateFn
from qiskit import Aer, execute
import numpy as np
from tqdm import tqdm
import time
import json

from vqe_vs_sum import unpack_twolocal, ising_model
from sample_permutations import my_gd, prepare_list_of_perms, mean_and_error
from main import permute_circuit


def c_2(q):
    """int cos^2 delta p(delta) d delta assuming that p(delta) is the normal
    distribution with variance q**2"""
    return np.exp(-(q * q) / 2)


def ms_angle_error(c2):
    """c_2 = int cos^2 delta p(delta) d delta
    c_1 = int sin delta cos delta p(delta) d delta, zero for even p(delta)
    c_0 = int sin^2 delta p(delta) d delta
    Assuming that p(delta) = p(-delta)"""

    c0 = 1 - c2
    x = np.array([[0, 1], [1, 0]])
    xx = np.kron(x, x)
    B_0 = c2 ** 0.5 * np.eye(4)
    B_1 = c0 ** 0.5 * xx
    return kraus_error([B_0, B_1])


def make_detuner(qs: np.array, coeff: float):
    def detuner(instruction: Instruction, qubits):
        c2 = c_2(float(instruction.params[0]) * coeff * qs[qubits[0], qubits[1]])
        noise_channel = ms_angle_error(c2)
        return noise_channel
    return detuner


def detune_circuit(circ, noisepass):
    dag = circuit_to_dag(circ)
    dag = noisepass.run(dag)
    return dag_to_circuit(dag)


def make_noisy_objective(h, circ, qs, coeff, shots=1024):
    """Objective function that mimics the behavior of a noisy circuit"""
    detuner = make_detuner(qs, coeff)
    noisepass = LocalNoisePass(func=detuner,
                               op_types=RXXGate,
                               method='append')
    backend = AerSimulator()
    qi = QuantumInstance(backend=backend, shots=shots)

    def f(x):
        circ_bnd = circ.assign_parameters(x)
        noisy_circ = detune_circuit(circ_bnd, noisepass)
        # print(noisy_circ)
        psi = CircuitStateFn(noisy_circ)
        op = StateFn(h, is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(op)
        sampler = CircuitSampler(qi).convert(expectation)
        return sampler.eval().real

    return f


def make_noisy_exact_objective(h, circ, qs, coeff):
    """Objective function that returns the exact expectation value
    for a circuit with appropriate MS channels in it"""
    detuner = make_detuner(qs, coeff)
    noisepass = LocalNoisePass(func=detuner,
                               op_types=RXXGate,
                               method='append')
    backend = AerSimulator(method='density_matrix')

    def f(x):
        circ_bnd = circ.bind_parameters(x)
        noisy_circ = detune_circuit(circ_bnd, noisepass)
        noisy_circ.save_density_matrix()
        result = execute(noisy_circ, backend).result()
        dm = result.data()['density_matrix']
        return np.trace(dm.data @ h.to_matrix()).real

    return f


def make_noiseless_objective(h, circ):
    backend = Aer.get_backend("statevector_simulator")

    def f(x):
        circ_bnd = circ.bind_parameters(x)
        result = execute(circ_bnd, backend).result()
        sv = result.get_statevector()
        return sv.expectation_value(h).real

    return f


def solve_best_then_use_on_perms(h: PauliSumOp,
                                 circ: QuantumCircuit,
                                 n_perms: int,
                                 qs: np.array,
                                 shots: int = 8192,
                                 tries_best: int = 20,
                                 tries_other: int = 5,
                                 maxiter: int = 50):
    perms, errorsums = prepare_list_of_perms(n_perms, circ, qs)

    energies_best = np.zeros(tries_best)
    stderrs_best = np.zeros(tries_best)
    optimal_points_best = np.zeros((tries_best, circ.num_parameters))
    energies_other = np.zeros((n_perms, tries_other))
    stderrs_other = np.zeros_like(energies_other)
    energies_other_bestpoint = np.zeros(n_perms)
    stderrs_other_bestpoint = np.zeros_like(energies_other_bestpoint)

    best_perm_index = np.argmin(errorsums)
    best_perm = perms[best_perm_index]
    h_best_perm = h.permute(list(best_perm))
    circ_best_perm = permute_circuit(circ, best_perm)
    objective = make_noisy_objective(h_best_perm, circ_best_perm, qs, shots)

    for i in tqdm(range(tries_best)):
        x0 = np.random.randn(circ.num_parameters) * 1e-1
        x = my_gd(objective, x0, niter=maxiter)
        # sol = vqe.compute_minimum_eigenvalue(h_best_perm)
        energies_best[i], stderrs_best[i] = mean_and_error(objective, x)
        # stderrs_best[i] = (sq - energies_best[i] ** 2) ** 0.5 / shots ** 0.5
        optimal_points_best[i, :] = x

    best_best_index = np.argmin(energies_best)
    best_point = optimal_points_best[best_best_index, :]
    print("Best energy: ", np.min(energies_best))
    print("Best parameters: ", best_point)
    print("Best index: ", best_best_index, np.argmin(energies_best))

    # solve other stuff using the best instance
    for i in tqdm(range(n_perms)):
        if i == best_perm_index:
            continue
        perm = perms[i]
        h_perm = h.permute(list(perm))
        circ_perm = permute_circuit(circ, perm)

        objective = make_noisy_objective(h_perm, circ_perm, qs, shots)

        energies_other_bestpoint[i], stderrs_other_bestpoint[i] = mean_and_error(objective, best_point)

        for j in range(tries_other):
            # print("reuse ", en_eval(best_point))
            x = my_gd(objective, best_point,
                      niter=maxiter)
            # sol = vqe.compute_minimum_eigenvalue(h_perm)
            energies_other[i, j], stderrs_other[i, j] = mean_and_error(objective, x)

            print("vqe ", energies_other[i, j])

    # values are converted to lists because numpy arrays are not json serializable
    out_dict = {"energies_best": energies_best.tolist(),
                "energies_other": energies_other.tolist(),
                "energies_other_bestpoint": energies_other_bestpoint.tolist(),
                "errorsums": errorsums.tolist(),
                "stderrs_best": stderrs_best.tolist(),
                "stderrs_other": stderrs_other.tolist(),
                "stderrs_other_bestpoint": stderrs_other_bestpoint.tolist(),
                "optimal_points_best": optimal_points_best.tolist(),
                "perms": perms.tolist()}
    return out_dict


if __name__ == "__main__":
    n_qubits = 8
    depth = 3
    n_perms = 20
    shots = 1024
    tries_best = 5
    tries_other = 1
    maxiter = 50

    circ = TwoLocal(n_qubits, ['ry'], 'rxx',
                    entanglement='linear',
                    reps=depth)
    circ = unpack_twolocal(circ)
    h = ising_model(n_qubits, 1, 1)

    rng_q = np.random.default_rng()
    qsquareds = abs(rng_q.normal(0, 1e-2, size=(n_qubits, n_qubits)))
    qsquareds = np.triu(qsquareds, 1) + np.triu(qsquareds, 1).T
    print(qsquareds)

    # in our model the correct thing seems to be the sum of squares!
    data = solve_best_then_use_on_perms(h, circ, n_perms, qsquareds**0.5, shots, tries_best,
                                        tries_other, maxiter)

    print(data)
    timestamp = int(time.time())
    with open(str(timestamp) + '.txt', 'w') as fp:
        json.dump(data, fp)
    print(timestamp)
    np.savetxt("qs_" + str(timestamp) + ".txt", qsquareds)
