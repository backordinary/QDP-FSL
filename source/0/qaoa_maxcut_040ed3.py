# https://github.com/soudy/bachelor-thesis/blob/85c0b1c03496fdae516b07da0d826e8d99616783/scripts/qaoa_maxcut.py
#! /usr/bin/env python3

import numpy as np
import networkx as nx
import argparse
import itertools
import time
from collections import defaultdict
from functools import partial

from qiskit import QuantumCircuit, Aer, execute
from qiskit.compiler import transpile, assemble

from projectq.ops import QubitOperator

from quantuminspire.credentials import enable_account, get_token_authentication
from quantuminspire.api import QuantumInspireAPI
from quantuminspire.qiskit import QI
from quantuminspire.qiskit.backend_qx import QuantumInspireBackend

from scipy.optimize import minimize
from noisyopt import minimizeSPSA, minimizeCompass

from typing import Optional, Dict, Any, List


DEFAULT_QI_API_URL = "https://api.quantum-inspire.com"


class QAOA:
    DEFAULT_SHOTS: int = 1024

    def __init__(self, G: nx.Graph,
                 n: int,
                 p: int = 1,
                 use_qi: bool = False,
                 qi_backend_type: str = ""):
        self.use_qi = use_qi
        self.G = G
        self.n = n
        self.p = p

        if use_qi:
            self.backend = QI.get_backend(qi_backend_type)
        else:
            self.backend = Aer.get_backend("qasm_simulator")

        self.current_step = {}
        self.optimization_steps = []

    def cost(self, params: List[float], shots: int = DEFAULT_SHOTS) -> float:
        cost = 0
        cost_hamiltonian = 0 * QubitOperator("")

        for i, j in self.G.edges:
            w = self.G[i][j]["weight"]
            cost_hamiltonian += 0.5 * w * (QubitOperator("") - QubitOperator(f"Z{i} Z{j}"))

        cost -= self.expval(cost_hamiltonian, params, shots=shots)

        return cost

    def record_cost_step(self, params: List[float], shots: int = DEFAULT_SHOTS) -> float:
        cost = self.cost(params, shots=shots)

        self.current_step["params"] = params
        self.current_step["cost"] = -cost

        print(self.current_step)

        self.optimization_steps.append(self.current_step)
        self.current_step = {}

        return cost

    def qaoa_circuit(self, params: List[float]) -> QuantumCircuit:
        circ = QuantumCircuit(self.n, self.n)

        for i in range(self.n):
            circ.h(i)

        circ.barrier()

        for i in range(self.p):
            # problem unitary
            for j, k in self.G.edges:
                circ.cnot(j, k)
                circ.rz(-params[i]*self.G[j][k]["weight"], k)
                circ.cnot(j, k)

            circ.barrier()

            # mixer unitary
            for j in range(self.n):
                circ.rx(2*params[i+1], j)

            circ.barrier()

        return circ

    def get_circuit_cqasm(self, params: List[float]) -> str:
        circ = self.qaoa_circuit(params)
        circ.measure(range(self.n), range(self.n))
        (experiment,) = assemble(transpile(circ, backend=self.backend),
                                 backend=self.backend).experiments

        return QuantumInspireBackend._generate_cqasm(
            experiment,
            full_state_projection=False
        )

    @staticmethod
    def _even_ones(binary: str, relevant_idxs: List[int]) -> int:
        integer = int(binary)
        ones_rel_idxs = 0
        for i in relevant_idxs:
            ones_rel_idxs += 10**i

        n_ones = str(integer + ones_rel_idxs).count("2")

        return 1 if n_ones % 2 == 0 else -1

    def probabilities(self, params: List[float],
                      shots: int = DEFAULT_SHOTS) -> Dict[str, int]:
        """
        Get the measurement probabilities for all basis states.

        Args:
            params: parameters for the QAOA circuit

        Returns:
            A dict mapping basis states to their probability.
        """
        circ = self.qaoa_circuit(params)
        circ.measure(range(self.n), range(self.n))

        counts = execute(circ, backend=self.backend, shots=shots) \
                    .result()                                     \
                    .get_counts(circ)

        results = {}
        for result, count in counts.items():
            results[result] = count/shots

        return results

    def expval_full(self, operator: QubitOperator, params: List[float],
                    shots: int = DEFAULT_SHOTS) -> float:
        """
        Calculate the expectation value of a Hamiltonian.

        Args:
            operator: Hamiltonian to calculate the expectation value of.
            params: parameters for the QAOA circuit
        Return:
            Expectation value of `operator` for QAOA state with parameters
                `params`
        """
        expval = 0

        # TODO: rewrite in terms of Qiskit PauliOp instead of requiring ProjectQ
        for term, coef in operator.terms.items():
            qubits_of_interest = []

            for qubit, op in term:
                qubits_of_interest.append(qubit)

            # if term is identity, add constant coefficient
            if term == ():
                expval += coef
                continue

            circ = self.qaoa_circuit(params)

            for qubit, op in term:
                qubits_of_interest.append(qubit)

                if op == "X":
                    circ.ry(-np.pi/2, qubit)
                elif op == "Y":
                    circ.rx(-np.pi/2, qubit)

            circ.measure(range(self.n), range(self.n))

            counts = execute(circ, backend=self.backend, shots=shots) \
                 .result()                                            \
                 .get_counts(circ)

            for result, count in counts.items():
                prob = count/shots
                parity = self._even_ones(result, qubits_of_interest)

                expval += parity * coef * prob

        return expval

    def expval(self, operator: QubitOperator, params: List[float],
               shots: int = DEFAULT_SHOTS) -> float:
        """
        Calculate the expectation value of a Hamiltonian. Only works for Pauli Z
        observables to make it only require one circuit execution.

        Args:
            operator: Hamiltonian to calculate the expectation value of.
            params: parameters for the QAOA circuit
        Return:
            Expectation value of `operator` for QAOA state with parameters
                `params`
        """
        expval = 0

        circ = self.qaoa_circuit(params)
        circ.measure(range(self.n), range(self.n))

        start_time = time.time()
        result = execute(circ, backend=self.backend, shots=shots).result()
        end_time = time.time()

        counts = result.get_counts(circ)

        self.current_step["quantum_total_time"] = end_time - start_time
        self.current_step["quantum_exec_time"] = result.to_dict()["results"][0]["time_taken"]
        self.current_step["quantum_wait_time"] = \
                self.current_step["quantum_total_time"] - self.current_step["quantum_exec_time"]

        # TODO: rewrite in terms of Qiskit PauliOp instead of requiring ProjectQ
        for term, coef in operator.terms.items():
            qubits_of_interest = []

            for qubit, op in term:
                qubits_of_interest.append(qubit)

            # if term is identity, add constant coefficient
            if term == ():
                expval += coef
                continue

            for result, count in counts.items():
                prob = count/shots
                parity = self._even_ones(result, qubits_of_interest)

                expval += parity * coef * prob

        return expval


def L(G: nx.Graph, z: List[int]) -> float:
    """
    Classical objective function for weighted Max-Cut.

    Args:
        G: weighted, undirected graph for which to calculate the
            objective function
        z: bit string that describes the bipartition

    Returns:
        Cost of solution `z` given graph `G`
    """
    cost = 0

    for i, j in G.edges:
        w = G[i][j]["weight"]
        cost += w*z[i]*(1 - z[j]) + w*z[j]*(1 - z[i])

    return cost


def init_graph() -> nx.Graph:
    n = 4
    V = np.arange(0, n, 1)
    E = [(0, 1, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]

    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)

    return G


def init_random_graph(d: int, n: int, seed: int, unit_weights: bool = False) -> nx.Graph:
    G = nx.random_regular_graph(d, n, seed=seed)

    for j, k in G.edges():
        if unit_weights:
            G[j][k]["weight"] = 1.0
        else:
            # set weight to random 2 decimal floating point between 0 and 4
            G[j][k]["weight"] = np.random.randint(0, 400) / 100

    return G

def grid_search_best_cuts(G: nx.Graph, n: int):
    results = defaultdict(list)
    zs = list(itertools.product([0, 1], repeat=n))

    for z in zs:
        c = L(G, z)
        results[c].append(z)

    max_cut = max(results, key=float)

    return max_cut, results[max_cut]


def set_qi_authentication(api_key: str, api_url: str, project_name: str = ""):
    enable_account(api_key)
    auth = get_token_authentication()
    QI.set_authentication(auth, api_url)


def init_params(p: int) -> List[float]:
    return np.random.uniform(low=0, high=2*np.pi, size=(2, p)).flatten("F")


def main(args):
    print(f"Running QAOA on Max-Cut with the following arguments:")
    for k, v in args.items():
        print(f"  {k:10}: {v}")

    np.random.seed(args["seed"])

    if args["random_graph"]:
        G = init_random_graph(args["random_graph_d"], args["random_graph_n"],
                              args["seed"],
                              unit_weights=args["random_graph_unit_weights"])
    else:
        G = init_graph()
    n = len(G.nodes)

    print("")
    print("Graph info:")
    print(f"  V = {G.nodes}")
    print(f"  |V| = {n}")
    print(f"  E = {G.edges(data=True)}")
    print(f"  |E| = {len(G.edges)}")

    if not args["skip_grid_search"]:
        max_cut, cuts = grid_search_best_cuts(G, n)
        print("")
        print("Grid search for best cuts:")
        print(f"  Max value: {max_cut}")
        print(f"  Max cut(s): {cuts}")
    else:
        max_cut, cuts = None, None

    params = init_params(args["layers"])

    configurable_kwargs = {"p": args["layers"]}
    qaoa = None
    if args["qi_backend_type"]:
        if args["qi_api_url"] == DEFAULT_QI_API_URL:
            # prod token
            token = "c15a7bdafcb71f7a3462c8515b23d64b7c3392d3"
        else:
            # staging token
            token = "ac4cd7fa1ce2967749bc9ae8d2ca26e78c8e0640"

        set_qi_authentication(token, args["qi_api_url"], project_name="qaoa-maxcut")
        qaoa = QAOA(G, n, use_qi=True, **configurable_kwargs,
                    qi_backend_type=args["qi_backend_type"])
    else:
        qaoa = QAOA(G, n, **configurable_kwargs)

    cost_fn = partial(qaoa.record_cost_step, shots=args["shots"])
    if args["optimizer"].lower() == "spsa":
        res = minimizeSPSA(cost_fn, bounds=[(0, 2*np.pi) for _ in params],
                           x0=params, niter=args["max_iter"],
                           paired=False)
    else:
        res = minimize(cost_fn, params, method=args["optimizer"],
                       options={"disp": True, "tol": 1e-6, "maxiter": args["max_iter"]})

    print(res)

    best_params = res.x
    probs = qaoa.probabilities(best_params, shots=args["solution_shots"])
    highest_prob = [int(b) for b in max(probs, key=probs.get)]
    real_objective = L(G, highest_prob)
    print(f"Best approximate solution = {highest_prob} with L({highest_prob}) = {real_objective}")

    # save results
    np.savez(args["output_file"], optimization_steps=qaoa.optimization_steps,
             optimization_result=res, final_probs=probs, final_cost=real_objective,
             G=G, max_cuts=(max_cut, cuts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAOA on Max-Cut",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--output-file", type=str, default="./qaoa_maxcut_out.npz",
        help="file to write results to")
    parser.add_argument(
        "--layers", type=int, default=1,
        help="number of QAOA layers")
    parser.add_argument(
        "--random-graph", action="store_true",
        help="generate a d-regular random graph with n nodes. requires "
             "--random-graph-d and --random-graph-n to be set")
    parser.add_argument(
        "--random-graph-unit-weights", action="store_true",
        help="use unit weights for random graph. generates random weights "
             "between 0 and 4 by default")
    parser.add_argument(
        "--random-graph-d", type=int, required=False,
        help="regularity d of random graph")
    parser.add_argument(
        "--random-graph-n", type=int, required=False,
        help="number of nodes of random graph")
    parser.add_argument(
        "--skip-grid-search", action="store_true",
        help="skip classical grid search for optimal cut(s)")
    parser.add_argument(
        "--optimizer", type=str, default="COBYLA",
        help="optimizer used")
    parser.add_argument(
        "--max-iter", type=int, default=30,
        help="max number of optimizer iterations")
    parser.add_argument(
        "--seed", type=int, default=0xc0ffee,
        help="seed for initializing parameters")
    parser.add_argument(
        "--shots", type=int, default=1024,
        help="number of shots used in quantum circuit execution")
    parser.add_argument(
        "--solution-shots", type=int, default=4096,
        help="number of shots used for measuring final solution")
    parser.add_argument(
        "--qi-backend-type", type=str, default="",
        help="Quantum Inspire back-end to use with Qiskit. if not provided, "
             "the Qiskit qasm_simulator back-end is used")
    parser.add_argument(
        "--qi-api-url", type=str, default=DEFAULT_QI_API_URL,
        help="Quantum Inspire API URL")
    args = parser.parse_args()

    if args.random_graph and (args.random_graph_d is None or args.random_graph_n is None):
        parser.error("--random-graph requires --random-graph-d and --random-graph-n")

    main(vars(args))
