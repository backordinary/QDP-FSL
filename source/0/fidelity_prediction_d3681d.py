# https://github.com/aleksey-uvarov/permutevqe/blob/6bfeeb35bc116a7d9b85fd6872ee5354a2852427/fidelity_prediction.py
"""In this file, we work on the following conjecture:
let single-qubit gates have depolarizing error p_j, and
two-qubit gates depolarizing error q_ij. The fidelity of the
final state w.r.t. the state prepared by the perfect circuit
is inversely correlated with the sum of p_j and q_ij taken
over the gates comprising the circuit. Hence, a good heuristic
to improve fidelity would be to pick a permutation that minimizes
this."""
import matplotlib.pyplot as plt

from main import *
import numpy as np
from qiskit_aer.backends import QasmSimulator
from qiskit import execute
import warnings
import json
import time
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':

    n_qubits = 10
    depth = 5
    p_magnitude = 1e-2
    q_magnitude = 1e-2
    qty_experiments = 50

    fidelities = np.zeros(qty_experiments, dtype='float')
    error_sums = np.zeros_like(fidelities)

    timestamp = int(time.time())

    for k in tqdm(range(qty_experiments)):

        ps = abs(np.random.randn(n_qubits) * p_magnitude)
        qs = abs(np.random.randn(n_qubits, n_qubits) * q_magnitude)
        qs = np.triu(qs, 1) + np.triu(qs, 1).T

        noise_model = NoiseModel()
        for j in range(n_qubits):
            noise_model.add_quantum_error(depolarizing_error(ps[j], 1),
                                          ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'], [j])
        for i in range(n_qubits):
            for j in range(n_qubits):
                noise_model.add_quantum_error(depolarizing_error(qs[i, j], 2),
                                              ['rxx'], [i, j])

        circ = TwoLocal(n_qubits, ['ry'], 'rxx',
                        entanglement='linear',
                        reps=depth)
        circ.save_density_matrix()

        parameters = np.random.rand(circ.num_parameters) * 2 * np.pi
        circ = circ.bind_parameters(parameters)

        backend_clean = Aer.get_backend('statevector_simulator')
        backend = QasmSimulator(method='density_matrix',
                                noise_model=noise_model)

        result = execute(circ, backend).result()
        result_clean = execute(circ, backend_clean).result()

        sv = result_clean.get_statevector()
        rho = result.data()['density_matrix']

        f = (sv.data.T.conj() @ rho.data @ sv.data).real

        error_rate_total = 0
        for gate in circ:
            defn = gate.operation.definition
            if 'data' not in dir(defn):
                continue
            for true_gate in defn.data:
                qubits = true_gate.qubits
                if len(qubits) == 2:
                    index_1, index_2 = qubits[0].index, qubits[1].index
                    error_rate_total += qs[index_1, index_2]
                elif len(qubits) == 1:
                    error_rate_total += ps[qubits[0].index]

        expt_data = {"n_qubits": n_qubits,
                     "depth": depth,
                     "p_magnitude": p_magnitude,
                     "q_magnitude": q_magnitude,
                     "fidelity": f,
                     "error_rate_total": error_rate_total}
        with open("data/experiment_data_{0:}_{1:}.json".format(timestamp, k), "w") as fp:
            json.dump(expt_data, fp)
        np.savetxt("data/ps_{0:}_{1:}.txt".format(timestamp, k), ps)
        np.savetxt("data/qs_{0:}_{1:}.txt".format(timestamp, k), qs)

        fidelities[k] = f
        error_sums[k] = error_rate_total

    np.savetxt('data/f_vs_fid_{0:}.txt'.format(timestamp), np.vstack((fidelities, error_sums)))

    plt.semilogy(error_sums, fidelities, 'o')
    plt.xlabel('Error sum')
    plt.ylabel('F')
    plt.grid(which='both')
    plt.savefig('data/f_vs_fid_{0:}.png'.format(timestamp), format='png',
                bbox_inches='tight', dpi=400)
    plt.show()


