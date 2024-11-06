# https://github.com/kevinab107/q-algos/blob/56cb8471e2b334af8f94256b51eba9fa128ea769/max_cut.py

from qiskit.aqua.algorithms.adaptive import QAOA
from qiskit import BasicAer

from qiskit.aqua.algorithms import VQE
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA
from qiskit.aqua import QuantumInstance
from data_models import get_maxcut_data_model
import numpy as np
import networkx as nx
from q_utils import get_compute_backend
from qiskit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    execute,
    transpile,
)
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_circuit_layout, plot_histogram
from q_utils import get_cicuit_layout
from collections import defaultdict


def get_configs():
    # TODO make the shots proportional to number of nodes
    shots_simulation = 100000
    shots_quantum = 2048
    p_repetitions = 1
    return {
        "p": p_repetitions,
        "shots_quantum": shots_quantum,
        "shots_simulation": shots_simulation,
    }


def get_operator(graph):

    weight_matrix = nx.to_numpy_array(graph)
    num_nodes = graph.number_of_nodes()
    operator_list = []
    z_p = np.zeros(num_nodes, dtype=np.bool)
    x_p = np.zeros(num_nodes, dtype=np.bool)
    weight_offset = 0
    for edge in graph.edges():
        if weight_matrix[edge] != 0:
            z_p[edge[0]] = True
            z_p[edge[1]] = True
            operator_list.append([0.5 * weight_matrix[edge], Pauli(z_p, x_p)])
            z_p[edge[0]] = False
            z_p[edge[1]] = False
            weight_offset -= 0.5 * weight_matrix[edge]

    return WeightedPauliOperator(paulis=operator_list), weight_offset


def get_maxcut_qubo(G, lagrange=None, weight="weight"):
    N = G.number_of_nodes()

    if lagrange is None:
        if G.number_of_edges() > 0:
            lagrange = G.size(weight=weight) * G.number_of_nodes() / G.number_of_edges()
        else:
            lagrange = 2

    Q = defaultdict(float)

    for u, v in G.edges():
        Q[(u, u)] = (1 - 2 * N) * lagrange
        Q[(v, v)] = (1 - 2 * N) * lagrange
        Q[(u, v)] = 2 * lagrange + G[u][v][weight]

    return Q


def solve_maxcut_vqe(graph, solve_type):
    backend = get_compute_backend(solve_type)
    operator, offset = get_operator(graph)
    vqe = VQE(
        operator,
        RYRZ(graph.number_of_nodes(), depth=3, entanglement="linear"),
        L_BFGS_B(maxfun=6000),
    )
    result = vqe.run(QuantumInstance(backend))
    return result


def build_cost_operator_circuit(qaoa_circuit, edges, gamma):
    for edge in edges:
        k = edge[0]
        l = edge[1]
        qaoa_circuit.cu1(-2 * gamma, k, l)
        qaoa_circuit.u1(gamma, k)
        qaoa_circuit.u1(gamma, l)
    return


def get_quantum_circuit(graph, gammas, betas):
    num_nodes = graph.number_of_nodes()
    edges = graph.edges()
    approx_level = get_configs()["p"]

    qaoa_circuit = QuantumCircuit(num_nodes)
    qaoa_circuit.h(range(num_nodes))
    qaoa_circuit.barrier()

    for depth in range(approx_level):
        build_cost_operator_circuit(qaoa_circuit, edges, gammas[depth])
        qaoa_circuit.barrier()
        qaoa_circuit.rx(2 * betas[depth], range(num_nodes))

    qaoa_circuit.measure_all()

    return qaoa_circuit


def get_eigen_values(state_bit):
    eigen_value_map = {0: -1, 1: 1}
    return eigen_value_map[state_bit]


def get_configuration_cost(state_bitstring, graph):

    edges = graph.edges()
    C = 0
    for edge in edges:
        k = edge[0]
        l = edge[1]
        k_eigen_val = get_eigen_values(int(state_bitstring[k]))
        l_eigen_val = get_eigen_values(int(state_bitstring[l]))

        w = graph[k][l]["weight"]
        C = C + w * (1 - k_eigen_val * l_eigen_val) / 2
    return C


def analyse_results(graph, counts):
    # counts = compute_result.get_counts()
    expectation_cost = 0
    max_cost = 0
    cost_dict = {}
    total_count = 0
    for state, count in counts.items():

        state_cost = get_configuration_cost(state, graph)
        cost_dict[state] = state_cost
        expectation_cost += count * state_cost
        total_count += count

    expectation_cost = expectation_cost / total_count
    return expectation_cost, cost_dict


def get_expectation_value(graph, counts):

    expectation_value, cost_dict = analyse_results(graph, counts)
    return expectation_value


def get_max_cost_state(graph, compute_result):

    expectation_value, cost_dict = analyse_results(graph, compute_result)
    max_cost_state = max(cost_dict, key=cost_dict.get)
    return (max_cost_state, cost_dict[max_cost_state])


def get_cost_dict(graph, compute_result):
    expectation_value, cost_dict = analyse_results(graph, compute_result)
    return cost_dict


def get_initial_angles():
    # optimizer = COBYLA(maxiter=500, tol=0.0001)
    approx_level = get_configs()["p"]
    params = np.random.rand(2 * approx_level)
    # initial_angles = optimizer.optimize(num_vars=2*approx_level, objective_function=objective_function, initial_point=params)
    res = minimize(objective_function, params, method="COBYLA", tol=1e-6)
    return res


def plot_result_graph(graph, result):
    color_conversion = lambda x: "blue" if x == 0 else "red"
    color_mapper = np.vectorize(color_conversion)
    labels = labels = nx.get_edge_attributes(graph, "weight")
    pos = nx.spring_layout(graph)
    nx.draw_networkx(
        graph,
        pos,
        node_color=color_mapper(result),
        with_labels=True,
        edge_labels=labels,
    )
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels, arrows=True)
    nx.draw_networkx_edges(graph, pos, width=4, edge_color="g", arrows=True)


def solve_maxcut_qaoa(graph):
    print("initialized backend")
    # operator, offset = get_operator(graph)
    # print("initialized operator")

    # # qaoa = QAOA(operator, p=1, optimizer=L_BFGS_B(maxfun=1000))
    # print("running the experinment")
    # result = qaoa.run(backend)
    # print("got the results")
    qc = get_quantum_circuit(graph, [1.9], [2])
    simulator = get_compute_backend("simulation")
    backend = get_compute_backend("quantum")  # currently set to IBMQ melbourne 16 qubit
    meas_job = execute(qc, simulator, shots=100000)
    result_sim = meas_job.result()
    counts_sim = result_sim.get_counts()

    quantum_job = execute(qc, backend)
    job_monitor(quantum_job)
    quantum_result = quantum_job.result()
    counts_quantum = quantum_result.get_counts()
    plot_histogram(
        [counts_sim, counts_quantum], legend=["simulator", "quantum_hardware"]
    )

    return counts_quantum


# # %%
# test_graph = get_maxcut_data_model()
# res = solve_maxcut_qaoa(test_graph)
# # %%
# # nx.draw(test_graph, pos=nx.spring_layout(test_graph))
# plot_result_graph(test_graph, res)
# get_cicuit_layout(qc, "ibmq_16_melbourne", 1)

# %%
