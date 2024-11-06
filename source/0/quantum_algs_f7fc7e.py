# https://github.com/V1ad0S/bachelor/blob/1e840a604de81ab7957eecd84b30ad04fb92b564/src/quantumoptimization/algorithms/quantum_algs.py
import logging
import logging.config

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.optimize import fmin_cobyla

from qiskit import QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram

from .helpers import Cut, edges_encoder, state_to_ampl_counts, logging_config


logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)



def qaoa_maxcut(G: nx.Graph, p: int = 1, theta_init: list = None,
                use_statevector: bool = False, hist: bool = False,
                max_iter: int = 1_000, shots: int = 1024) -> Cut:

    solution = get_optimal(G, p, theta_init, shots,
                           use_statevector, max_iter=max_iter)
    # if not solution.success:
    #     logger.warning(f'QAOA: {solution.message}')
    #     logger.warning(f'QAOA: Optimal parameters NOT founded!')
    # else:
    #     logger.info('QAOA: Optimal parametres founded!')

    # print(solution)

    backend = Aer.get_backend('aer_simulator')

    qc_result = _create_qaoa_circuit(G, solution, measure=True)
    counts = execute(qc_result, backend=backend, shots=10_000, seed=10).result().get_counts()

    logger.info('QAOA: results obtained!')

    top_counts_keys = sorted(counts, key=(lambda key: counts[key]), reverse=True)[:100]

    if hist:
        _hist_plot(counts, top_counts_keys, G)

    bitstring_res = _get_max_cut(top_counts_keys, G)

    cut = bitstring2cut(G, bitstring_res)
    return cut

def _hist_plot(counts: dict, top_counts_keys: list, G: nx.Graph) -> None:
    top_counts = {k: counts[k] for k in top_counts_keys[:15]}

    fig, ax = plt.subplots(figsize=(20, 8))
    plot_histogram(top_counts, ax=ax)

    labels = [bstr.get_text() for bstr in ax.get_xticklabels()]
    x_values = [-maxcut_objective(x[::-1], edges_encoder(G)) for x in labels]
    ax.set_xticklabels(x_values)

    return ax

def _get_max_cut(counts: list, G: nx.Graph) -> str:
    return counts[0][::-1]

    # TODO: need to remove bruteforce
    max_cut = [0, None]
    edges = edges_encoder(G)

    for bstr in counts:
        cut_size = maxcut_objective(bstr[::-1], edges)
        if cut_size < max_cut[0]:
            max_cut[0] = cut_size
            max_cut[1] = bstr[::-1]

    return max_cut[1]

def bitstring2cut(G: nx.Graph, bitstring: str) -> Cut:
    cut = Cut(G)
    nodes_encoder = dict(zip(range(G.number_of_nodes()), G.nodes))
    distributor = {
        '0': cut.add_left,
        '1': cut.add_right,
    }
    for i, val in enumerate(bitstring):
        distributor[val](nodes_encoder[i])
    return cut

def get_optimal(G: nx.Graph, p: int, theta_init: list, shots: int = 512,
                use_statevector: bool = False, max_iter: int = 1_000):
    expectation = _get_expectation(G, shots, statevector=use_statevector)

    if not theta_init:
        theta_init = [1.] * 2 * p

    logger.info('QAOA: Optimization started')
    res = fmin_cobyla(expectation, theta_init, cons=lambda _: 1, rhobeg=0.5,
                      rhoend=1e-2, maxfun=max_iter, disp=0)
    return res

def maxcut_objective(x: str, edges: list[tuple]) -> int:
    obj = 0

    for v1, v2 in edges:
        if x[v1] != x[v2]:
            obj -= 1

    return obj

def compute_maxcut_energy(counts: dict, edges: list[tuple]) -> float:
    avg = 0
    sum_count = 0

    for bitstring, count in counts.items():
        obj = maxcut_objective(bitstring[::-1], edges)
        avg += obj * count
        sum_count += count

    return avg / sum_count

def compute_maxcut_energy_statevector(sv: np.ndarray, edges: list[tuple]) -> float:
    """
    Compute objective from statevector
    Too slow for large number of qubits. 
    """
    counts = state_to_ampl_counts(sv)
    energy = 0

    for bitstring, count in counts.items():
        obj = maxcut_objective(bitstring[::-1], edges)
        energy += obj * np.abs(count)**2

    return energy

def _get_expectation(G: nx.Graph, shots, statevector: bool = False, seed: int = 10):
    if statevector:
        backend = Aer.get_backend('statevector_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')

    edges = edges_encoder(G)

    def execute_circ(theta):
        qc = _create_qaoa_circuit(G, theta, measure=True)
        counts = execute(qc, backend=backend, seed_simulator=seed,
                         shots=shots).result().get_counts()
        return compute_maxcut_energy(counts, edges)
    
    def execute_circ_sv(theta):
        qc = _create_qaoa_circuit(G, theta, measure=False) # don't need to measure
        sv = execute(qc, backend=backend, seed_simulator=seed,
                     shots=shots).result().get_statevector()
        return compute_maxcut_energy_statevector(sv, edges)

    if statevector:
        return execute_circ_sv
    return execute_circ

def _create_qaoa_circuit(G: nx.Graph, theta: list, measure: bool = True) -> None:
    assert len(theta) % 2 == 0

    nqubits = G.number_of_nodes()
    edges = edges_encoder(G)

    p = len(theta) // 2
    betas = theta[:p]
    gammas = theta[p:]

    qcircuit = QuantumCircuit(nqubits)
    _init_state(qcircuit, nqubits)
    qcircuit.barrier()

    for beta, gamma in zip(betas, gammas):
        _add_problem_unitary(qcircuit, gamma, edges)
        qcircuit.barrier()
        _add_mixing_unitary(qcircuit, beta, nqubits)
    
    if measure:
        qcircuit.measure_all()

    return qcircuit

def _init_state(qcircuit: QuantumCircuit, nqubits: int) -> None:
    for i in range(nqubits):
        qcircuit.h(i)

def _add_mixing_unitary(qcircuit: QuantumCircuit, beta: float, nqubits: int) -> None:
    for i in range(nqubits):
        qcircuit.rx(2 * beta, i)

def _add_problem_unitary(qcircuit: QuantumCircuit, gamma: float,
                         edges: list[tuple]) -> None:
    for v1, v2 in edges:
        qcircuit.rzz(2 * gamma, v1, v2)