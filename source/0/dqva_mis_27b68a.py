# https://github.com/Quantum-Software-Tools/quantum-constrained-optimization/blob/7ebddcfd058bc2080e5ad25581a653abe962dcaf/qcopt/dqva_mis.py
import copy

import numpy as np
import networkx as nx
import qiskit
from qiskit import Aer
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from qcopt.ansatz import dqva
from qcopt.utils import graph_funcs
from qcopt.utils import helper_funcs


def solve_mis(
    init_state,
    G,
    P=1,
    m=1,
    mixer_order=None,
    threshold=1e-5,
    cutoff=1,
    sim="statevector",
    shots=8192,
    verbose=0,
    threads=0,
):
    """
    Find the MIS of G using the dynamic quantum variational ansatz (DQVA),
    this ansatz has the same structure as QLS but does not include QLS's
    parameter limit
    """

    # Initialization
    backend = Aer.get_backend("aer_simulator_statevector", max_parallel_threads=threads)
    if sim == "cloud":
        raise Exception("NOT YET IMPLEMENTED!")
    elif sim not in ["statevector", "qasm"]:
        raise Exception("Unknown simulator:", sim)

    # Select and order for the partial mixers
    if mixer_order == None:
        cur_permutation = list(np.random.permutation(list(G.nodes)))
    else:
        cur_permutation = mixer_order

    history = []

    # This is the function which scipy.minimize will optimize
    def f(params):
        # Generate a DQVA circuit
        circ = dqva.gen_dqva(
            G,
            P,
            params=params,
            init_state=cur_init_state,
            barriers=0,
            decompose_toffoli=1,
            mixer_order=cur_permutation,
            verbose=0,
        )

        if sim == "qasm":
            circ.measure_all()
        elif sim == "statevector":
            circ.save_statevector()

        # Compute the cost function
        result = qiskit.execute(circ, backend=backend, shots=shots).result()
        if sim == "statevector":
            probs = Statevector(result.get_statevector(circ)).probabilities_dict(decimals=5)
        elif sim == "qasm":
            probs = {key: val / shots for key, val in result.get_counts(circ).items()}

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        # print('Expectation value:', avg_cost)
        return -avg_cost

    # Begin outer optimization loop
    best_indset = init_state
    best_init_state = init_state
    cur_init_state = init_state
    best_params = None
    best_perm = copy.copy(cur_permutation)

    # Randomly permute the order of mixer unitaries m times
    for mixer_round in range(1, m + 1):
        mixer_history = []
        inner_round = 1
        new_hamming_weight = helper_funcs.hamming_weight(cur_init_state)

        # Attempt to improve the Hamming weight until no further improvements can be made
        while True:
            if verbose:
                print(
                    "Start round {}.{}, Initial state = {}".format(
                        mixer_round, inner_round, cur_init_state
                    )
                )

            # Begin Inner variational loop
            # num_params = P * ((len(G.nodes()) - hamming_weight(cur_init_state)) + 1)
            num_params = P * (len(G.nodes()) + 1)
            if verbose:
                print("\tNum params =", num_params)
            # Important to start from random initial points
            init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=num_params)
            if verbose:
                print("\tCurrent Mixer Order:", cur_permutation)

            out = minimize(f, x0=init_params, method="COBYLA")

            opt_params = out["x"]
            opt_cost = out["fun"]
            if verbose:
                print("\tOptimal cost:", opt_cost)

            # Get the results of the optimized circuit
            opt_circ = dqva.gen_dqva(
                G,
                P,
                params=opt_params,
                init_state=cur_init_state,
                barriers=0,
                decompose_toffoli=1,
                mixer_order=cur_permutation,
                verbose=0,
            )

            if sim == "qasm":
                opt_circ.measure_all()
            elif sim == "statevector":
                opt_circ.save_statevector()

            result = qiskit.execute(opt_circ, backend=backend, shots=shots).result()
            if sim == "statevector":
                probs = Statevector(result.get_statevector(opt_circ)).probabilities_dict(decimals=5)
            elif sim == "qasm":
                probs = {key: val / shots for key, val in result.get_counts(opt_circ).items()}

            # Select the top [cutoff] bitstrings
            top_counts = sorted(
                [(key, val) for key, val in probs.items() if val > threshold],
                key=lambda tup: tup[1],
                reverse=True,
            )[:cutoff]

            # Check if we have improved the Hamming weight
            best_hamming_weight = helper_funcs.hamming_weight(best_indset)
            better_strs = []
            for bitstr, prob in top_counts:
                this_hamming = helper_funcs.hamming_weight(bitstr)
                if (
                    graph_funcs.is_indset(bitstr, G, little_endian=True)
                    and this_hamming > best_hamming_weight
                ):
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # Save current results to history
            inner_history = {
                "mixer_round": mixer_round,
                "inner_round": inner_round,
                "cost": opt_cost,
                "function_evals": out["nfev"],
                "init_state": cur_init_state,
                "mixer_order": copy.copy(cur_permutation),
                "num_params": num_params,
                "opt_params": opt_params,
                "P": P,
                "better_strs": better_strs,
            }
            mixer_history.append(inner_history)

            # If no improvement was made, break and go to next mixer round
            if len(better_strs) == 0:
                print(
                    "\tNone of the measured bitstrings had higher Hamming weight than:", best_indset
                )
                break

            # Otherwise, save the new bitstring and repeat
            best_indset, new_hamming_weight = better_strs[0]
            best_init_state = cur_init_state
            best_params = opt_params
            best_perm = copy.copy(cur_permutation)
            cur_init_state = best_indset
            print(
                "\tFound new independent set: {}, Hamming weight = {}".format(
                    best_indset, new_hamming_weight
                )
            )
            inner_round += 1

        # Save the history of the current mixer round
        history.append(mixer_history)

        # Choose a new permutation of the mixer unitaries
        cur_permutation = list(np.random.permutation(list(G.nodes)))

    print("\tRETURNING, best hamming weight:", new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm, history
