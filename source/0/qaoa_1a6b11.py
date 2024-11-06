# https://github.com/jcking1034/qiskit-algorithms/blob/b405ff8b6e63bd55d3e33cd8d0c378ba42816d2c/qaoa.py
"""
This is an implementation of the quantum approximation algorithm as proposed in [the original paper](https://arxiv.org/abs/1411.4028) by Farhi, Goldstone, and Gutmann.
Our implementation draws much guidance from [Google's tutorial](https://quantumai.google/cirq/tutorials/qaoa) and improves it by allowing for a much wider range of graphs to be operated on. This is only possible since our implementation is simulated on a classical computer (where the relative "position" and "connections" between the simulated qubits are not taken into account).
"""

import random
import sys
import time  # to sample a random partition to check against QAOA

import cirq  # for quantum simulation
import cirq_google  # F_REMOVED to set working device below
import matplotlib.pyplot as plt  # F_REMOVED: to draw examples along the way
import networkx as nx  # to construct and work with graphs graphs
import numpy as np  # for general numberical manipulations
# for working with alpha and beta, the paramters of the algorithm
import sympy
import qiskit
from qiskit import IBMQ, Aer, QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from scipy.optimize import minimize

working_device = cirq_google.Bristlecone


USE_IBMQ = True
SIMULATOR_NOISE = True
VERBOSE = True

if USE_IBMQ or SIMULATOR_NOISE:
    IBMQ.save_account("YOUR_KEY_HERE")


GRAPHICAL_DISPLAY = False
NUM_SHOTS = 20


def is_not_proper_adjacency_matrix(l):
    if not (isinstance(l, list) and isinstance(l[0], list)):
        print(
            "The input read is 1 dimensional or has no dimension at all. We need a matrix."
        )
        return True

    first_row_len = len(l[0])
    if first_row_len >= 16:
        print(
            "you have more then 16 nodes in your graph, we will run the program for you, but note that it might take ridicoulusly long"
        )
    # check that it's a matrix and that it's square
    for row in l:
        if len(row) != first_row_len:
            print(
                "the row",
                row,
                "length is not",
                first_row_len,
                "as it should be. This is not a matrix, try again!",
            )
            return True
    for i in range(first_row_len):
        for j in range(first_row_len):
            if i == j and l[i][j] != 0:
                print(
                    "The node",
                    i,
                    "has an edge of weight",
                    l[i][j],
                    "with itself. In the context of this problem, it doesn't make sense.",
                )
                return True
            if l[i][j] != l[j][i]:
                print(
                    "The matrix at position",
                    (i, j),
                    "does not equal the matrix in position",
                    (j, i),
                    ". Remember, this graph must be undirected.",
                )
                return True
    return False


def msinumeric(x):
    if len(x) == 0:
        return False
    for char in x:
        if char not in "0,1,2,3,4,5,6,7,8,9,.,-".split(","):
            return False
    return True


def rem_newline_and_return_self(lst):
    y = [x for x in lst if msinumeric(x)]
    # print(y)
    return y


def generate_problem_instance(n):
    """
    Whip-up the problem instance. I'll take this as a given, and once everyhting else is working fine and dandy, I'll come back and change this.
    """
    with open(f"adjacency_matrices/{n}", "r") as f:
        lines = f.readlines()
        l = [
            [float(num) for num in rem_newline_and_return_self(line.strip().split(" "))]
            for line in lines
        ]
    # print(l)
    # for now, it is the user's responsibility to create a proper adjacency matrix
    if is_not_proper_adjacency_matrix(l):
        print(
            "As the last print indicated, your adjacency matrix is not proper.\n Exiting the program..."
        )
        exit(1)

    # Populate a networkx graph with working_qubits as nodes.
    working_graph = nx.Graph()
    for i in range(len(l[0])):  # add as many nodes as the adjacency matrix requests
        working_graph.add_node(i)

    # Pair up all neighbors with random weights in working_graph.
    for i in range(len(l[0])):
        # we must only iterate up to the diagonal since the matrix is symetric about it
        for j in range(i, len(l[0])):
            if l[i][j] > 0:
                working_graph.add_edge(i, j, weight=l[i][j])
    num_nodes = working_graph.number_of_nodes()
    if GRAPHICAL_DISPLAY:
        plt.title(f"Your original graph, for size {num_nodes}:")
        nx.draw_spring(working_graph, node_size=1000, with_labels=True)
        # nx.draw_networkx_edge_labels(
        #   working_graph, pos=nx.spring_layout(working_graph))
        # print(working_graph.edges(data=True))
        plt.show()
    return working_graph, num_nodes


def construct_circuit(working_graph, alpha, beta):
    """construct circuit from working graph"""

    # Symbols for the rotation angles in the QAOA circuit.

    qaoa_circuit = cirq.Circuit(
        # Prepare uniform superposition on working_qubits == working_graph.nodes
        cirq.H.on_each(working_graph.nodes()),
        # Do ZZ operations between neighbors u, v in the graph. Here, u is a qubit,
        # v is its neighboring qubit, and w is the weight between these qubits.
        (
            cirq.ZZ(u, v) ** (alpha * w["weight"])
            for (u, v, w) in working_graph.edges(data=True)
        ),
        # Apply X operations along all nodes of the graph. Again working_graph's
        # nodes are the working_qubits. Note here we use a moment
        # which will force all of the gates into the same line.
        cirq.Moment(cirq.X(qubit) ** beta for qubit in working_graph.nodes()),
        # All relevant things can be computed in the computational basis.
        (cirq.measure(qubit) for qubit in working_graph.nodes()),
    )
    return qaoa_circuit


def estimate_cost(graph, samples):
    """
    Estimate the cost function of the QAOA on the given graph using the
    provided computational basis bitstrings.
    I STILL HAVE NO IDEA WHAT THIS DOES
    *this may be something which I shouldn't do? I think it for graphs with no definite edge weights and whatnot?*
    """
    cost_value = 0.0

    # Loop over edge pairs and compute contribution.
    for u, v, w in graph.edges(data=True):
        u_samples = samples[str(u)]
        v_samples = samples[str(v)]

        # Determine if it was a +1 or -1 eigenvalue.
        u_signs = (-1) ** u_samples
        v_signs = (-1) ** v_samples
        term_signs = u_signs * v_signs

        # Add scaled term to total cost.
        term_val = np.mean(term_signs) * w["weight"]
        cost_value += term_val

    return -cost_value


def search_over_parmspace_for_best_alpha_beta(sim, qaoa_circuit, grid_size):
    # Set the grid size = number of points in the interval [0, 2π).

    exp_values = np.empty((grid_size, grid_size))
    par_values = np.empty((grid_size, grid_size, 2))

    for i, alpha_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
        for j, beta_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
            samples = sim.sample(
                qaoa_circuit,
                params={alpha: alpha_value, beta: beta_value},
                repetitions=20000,
            )
            exp_values[i][j] = estimate_cost(working_graph, samples)
            par_values[i][j] = alpha_value, beta_value
    if GRAPHICAL_DISPLAY:
        plt.title("Heatmap of QAOA Cost Function Value")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")
        plt.imshow(exp_values)
        plt.show()
    return exp_values, par_values


def visualise_cut(S_partition, working_graph, plt_title_string):
    """Plot and output the graph cut information.
    could be eventually removed
    """

    # Generate the colors.
    coloring = []
    for node in working_graph:
        if node in S_partition:
            coloring.append("blue")
        else:
            coloring.append("red")

    # Get the weights
    edges = working_graph.edges(data=True)
    weights = [w["weight"] for (u, v, w) in edges]
    if GRAPHICAL_DISPLAY:
        plt.title(plt_title_string)
        nx.draw_circular(
            working_graph,
            node_color=coloring,
            node_size=1000,
            with_labels=True,
            width=weights,
        )
        plt.show()
    size = nx.cut_size(working_graph, S_partition, weight="weight")
    # print(f'Cut size: {size}')
    return size


def create_qaoa_circuit(G, theta):
    n = len(G.nodes())
    p = len(theta) // 2
    qc = QuantumCircuit(n)

    beta = theta[:len(theta) // 2]
    gamma = theta[len(theta) // 2:]

    # initial_state
    for i in range(0, n):
        qc.h(i)

    for irep in range(0, p):

        # problem unitary
        for pair in list(G.edges()):
            start = pair[0]
            stop = pair[1]
            qc.rzz(2 * gamma[irep], start, stop)

        # mixer unitary
        for i in range(0, n):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc


def get_expectation(G, shots=512):
    if SIMULATOR_NOISE:
        provider = IBMQ.load_account()
        noise_backend = provider.backend.ibmq_quito
        noise_model = NoiseModel.from_backend(noise_backend)
        backend = QasmSimulator(noise_model=noise_model)
    else:
        backend = Aer.get_backend("qasm_simulator")
    # if USE_IBMQ:
    #     provider = IBMQ.load_account()
    #     backend = provider.backend.ibmq_quito
    # elif SIMULATOR_NOISE:
    #     provider = IBMQ.load_account()
    #     noise_backend = provider.backend.ibmq_quito
    #     noise_model = NoiseModel.from_backend(noise_backend)
    #     backend = QasmSimulator(noise_model=noise_model)
    # else:
    #     backend = QasmSimulator()
    # backend = Aer.get_backend("qasm_simulator")
    backend.shots = shots

    def execute_circuit(theta):
        # print("execute_circuit", backend)

        qc = create_qaoa_circuit(G, theta)
        transpiled = qiskit.transpile(qc, backend)
        result = backend.run(transpiled, seed_simulator=10, nshots=512).result()
        # if VERBOSE:
        #     print(result)
        counts = result.get_counts()

        # Calculate expectation
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            # Get maxcut obj
            obj = 0
            for i, j in G.edges():
                if bitstring[i] != bitstring[j]:
                    obj -= 1

            avg += obj * count
            sum_count += count

        return avg / sum_count

    return execute_circuit


def run_qaoa(G):
    expectation = get_expectation(G)
    res = minimize(expectation, [1.0, 1.0], method="COBYLA")
    if VERBOSE:
        print(res)

    if USE_IBMQ:
        provider = IBMQ.load_account()
        backend = provider.backend.ibmq_quito
    elif SIMULATOR_NOISE:
        provider = IBMQ.load_account()
        noise_backend = provider.backend.ibmq_quito
        noise_model = NoiseModel.from_backend(noise_backend)
        backend = QasmSimulator(noise_model=noise_model)
    else:
        backend = QasmSimulator()
    # backend = Aer.get_backend("aer_simulator")
    # print("Backend for final result:", backend)

    qc_res = create_qaoa_circuit(G, res.x)
    transpiled = qiskit.transpile(qc_res, backend)
    result = backend.run(transpiled, seed_simulator=10, shots=1024).result()
    print("Final result:", result)
    counts = result.get_counts()

    best_qaoa_S_partition = set()
    best_qaoa_cut_size = -np.inf
    for k, v in counts.items():
        one_qubits = set()
        for i, x in enumerate(k):
            if x == "1":
                one_qubits.add(i)
        print(one_qubits, working_graph.nodes())

        S_partition = set()
        T_partition = set()
        for node in working_graph:
            if node in one_qubits:
                # If a one was measured add node to S partition.
                S_partition.add(node)
            else:
                # Otherwise a zero was measured so add to T partition.
                T_partition.add(node)

        cut_size = nx.cut_size(working_graph, S_partition, T_partition, weight="weight")

        # If you found a better cut update best_qaoa_cut variables.
        if cut_size > best_qaoa_cut_size:
            best_qaoa_cut_size = cut_size
            best_qaoa_S_partition = S_partition

    return best_qaoa_S_partition, best_qaoa_cut_size


def get_best_random_cut_out_of(num_cuts):
    best_random_S_partition = set()
    best_random_cut_size = -9999

    # Randomly build candidate sets.
    for i in range(num_cuts):
        S_partition = set()
        T_partition = set()
        for node in working_graph:
            if random.random() > 0.5:
                # If we flip heads add to S.
                S_partition.add(node)
            else:
                # Otherwise add to T.
                T_partition.add(node)

        cut_size = nx.cut_size(working_graph, S_partition, T_partition, weight="weight")

        # If you found a better cut update best_random_cut variables.
        if cut_size > best_random_cut_size:
            best_random_cut_size = cut_size
            best_random_S_partition = S_partition
    return best_random_S_partition


if __name__ == "__main__":
    qaoa_cut_sizes = []
    random_cut_sizes = []
    circ_runtimes = []
    tot_runtimes = []

    try:
        min_size = int(sys.argv[1])
        max_size = int(sys.argv[2])
    except Exception:
        print("Usage: python qaoa.py [MIN SIZE] [MAX SIZE]")
        exit(1)

    for n in range(min_size, max_size + 1):
        start_tot_time = time.perf_counter()

        alpha = sympy.Symbol("alpha")
        beta = sympy.Symbol("beta")
        sim = cirq.Simulator()

        working_graph, num_nodes = generate_problem_instance(n)

        start = time.perf_counter()
        best_qaoa_S_partition, best_qaoa_cut_size = run_qaoa(working_graph)
        stop = time.perf_counter()

        circ_runtimes.append(stop - start)

        print("best of QAOA:", best_qaoa_S_partition, best_qaoa_cut_size)

        # do random guess for comparison
        best_random_S_partition = get_best_random_cut_out_of(NUM_SHOTS)

        # print the results

        # print('-----QAOA-----')
        qaoa_cut_sizes.append(
            visualise_cut(
                best_qaoa_S_partition,
                working_graph,
                f"The cut as found by QAOA for size {num_nodes}",
            )
        )
        # print('\n\n-----RANDOM-----')
        random_cut_sizes.append(
            visualise_cut(
                best_random_S_partition,
                working_graph,
                f"The cut as found by random searching for size {num_nodes}",
            )
        )
        tot_runtimes.append(time.perf_counter() - start_tot_time)

    precentege_in_circ = []
    qaoa_rel_to_rand = []
    print(f"\tFINAL RESULTS FOR CIRCUITS [{min_size},{max_size+1})")
    print("-------------------------------------------------------------------------\n")
    for n in range(max_size - min_size + 1):
        precentege_in_circ.append(circ_runtimes[n] / tot_runtimes[n])
        qaoa_rel_to_rand.append(qaoa_cut_sizes[n] / random_cut_sizes[n])
        print(
            f"For n={n+min_size} we have a total circuit runtime {circ_runtimes[n]} and total runtime {tot_runtimes[n]} which means we spent {precentege_in_circ[n]} of the time runnign the circuit"
        )
        print(
            f"\tQAOA found a best cut of size {qaoa_cut_sizes[n]} while random search found a best cut of size {random_cut_sizes[n]}, for relative score of {qaoa_rel_to_rand[n]}."
        )
    print("\n\tFINAL RESULTS:")
    print("-------------------------------------------------------------------------")
    print(
        f"On average {np.average(np.array(precentege_in_circ))} of the time in circuits, and qaoa's relative preformence to random was {np.average(np.array(qaoa_rel_to_rand))}"
    )
