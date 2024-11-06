# https://github.com/dowusu-antwi/qaoa-research/blob/8f2d6c1c1bd5b334bd10f3b02f12f1de50285236/qaoa_sim.py
#!/usr/bin/env python3

"""
QAOA Noiseless + Noisy Simulations

(Based on Combinatorial optimization with QAOA, see: https://qiskit.org/textbook/ch-applications/qaoa.html)

author: dowusu
"""

###############################################################################
# Useful imports...
###############################################################################
import sys
from pickle import dump, load

# Imports necessary graph and plotting tools
print("Importing plotting tools (from numpy, networkx, matplotlib, etc.)...")
import numpy as np
from random import random
from math import pi
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from time import time
print("Plotting tools imported.")

# Imports necessary qiskit tools to build and execute on IBMQ
print("Importing standard qiskit tools...")
from qiskit import Aer, IBMQ, transpile
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, circuit_drawer
print("Qiskit tools imported.")


# Imports noise modeling tools
print("Importing qiskit noise tools...")
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
print("Qiskit noise tools imported.")

# Imports error mitigation tools
print("Importing qiskit (mitiq) error mitigation tools...")
from mitiq import zne
from mitiq.zne.inference import PolyFactory
from mitiq.zne.scaling import fold_gates_from_left, fold_gates_from_right
print("Mitiq error mitigation tools imported.")

## Useful constants... (e.g., text separators for printing)
FOLDING_BASIS_1Q = ["rx", "rz"]
FOLDING_BASIS_2Q = ["cx"]
QISKIT_BASIS_1Q = ["p", "rx"]
QISKIT_BASIS_2Q = ["cp"]
SHOTS = 10000           # number of execution repetitions, for sampling
EDGE_PROBABILITY = 0.5  # for building random connectivity graph (erdos-renyi)
FIXED_WIDTH = 5         # circuit width for rounds calculation
SEP = "=" * 100
SEPS = {'enter': "=" * 10 + " ENTERING FUNCTION " + "=" * 10,
        'exit': "=" * 10 + " EXITING FUNCTION " + "=" * 10}
VERBOSE = 0 # 0 - no prints, 1 - basic prints, 2 - all prints
PLOT = 0 # 0 - no plotting, 1 - plotting

###############################################################################
# Useful functions...
###############################################################################

def show(descr=None):
    """
    Plots data and shows figure (maximized) for given description.
    """
    # Shows plot live, using standard plt.show().
    if descr == "live":
        figure = plt.gcf()
        plt.show()
        #manager = figure.canvas.manager
        #manager.window.showMaximized() # These lines are to make sure that the
        #figure.tight_layout()          #  maximized plot is saved.
        return figure

    # Otherwise, asks for verbose output (i.e., showing plot v. not showing). 
    if descr:
        answer = input("Show '%s'? (y): " % descr)
        if answer != 'y':
            return
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.tight_layout()


def build_butterfly(num_qubits):
    """
    Builds butterfly graph representing qubit connectivity (see QAOA qiskit
      tutorial.
    """
    nodes = np.arange(num_qubits)
    edges = [(0, 1), (0, 2), (1,2), (3,2), (3,4), (4,2)]
    edge_weights = [1.0] * len(edges)
    weighted_edges = [edge + (edge_weights[index],) for index, edge in 
                      enumerate(edges)]
    butterfly = nx.Graph()
    butterfly.add_nodes_from(nodes)
    butterfly.add_weighted_edges_from(weighted_edges)
    return butterfly


def build_param_space(step):
    """
    Gets 2D grids of gamma and beta parameters, for parameter searching.
    """
    gamma_axis, beta_axis = np.arange(0, pi, step), np.arange(0, pi, step)
    gamma_grid, beta_grid = np.meshgrid(gamma_axis, beta_axis)
    return gamma_grid, beta_grid


def get_optimal_butterfly():
    """
    Evaluates expectation, using grid search to find optimal parameters
      maximizing its value.
    """
    # Evaluates expectation via considering connectivity of butterfly graph,
    #  using hard-coded analytic expectation.
    step = 0.1
    gamma_grid, beta_grid = build_param_space(step)
    expectation = 3 - (np.sin(2 * beta_grid)**2 * np.sin(2 * gamma_grid)**2 \
                - 0.5 * np.sin(4 * beta_grid) * np.sin(4 * gamma_grid)) \
                * (1 + np.cos(4 * gamma_grid)**2)
    
    # Performs grid search (with numpy) on expectation, finding args-max
    max_expectation = np.max(expectation)
    optimum_indices = np.where(expectation == max_expectation)
    gamma_opt, beta_opt = [float(index * step) for index in optimum_indices]
    print("Optimal params (gamma, beta): %.02f, %.02f" % (gamma_opt, beta_opt))
    return gamma_opt, beta_opt, expectation


def plot_expectation_3D(expectation):
    """
    Plots graph of expectation value in 3D, as a function of gamma and
      beta parameters.
    """
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    
    zmin, zmax = 1, 4
    COLORMAP = cm.coolwarm
    LINEWIDTH = 0
    ANTIALIAS = False  # should force opaque (irrelevant on jupyter...)
    
    axis.set_title("Expectation Value")
    axis.set_xlabel("Gamma")
    axis.set_ylabel("Beta")
    axis.set_zlabel("Expectation")
    axis.set_zlim(zmin, zmax)
    
    # Sets zaxis tick locator (linear with 3 ticks?) and tick label format
    axis.zaxis.set_major_locator(LinearLocator(numticks=3))
    axis.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    step = 0.1
    gamma_grid, beta_grid = build_param_space(step)
    surface = axis.plot_surface(gamma_grid, beta_grid, expectation,
                                cmap=COLORMAP, 
                                linewidth=LINEWIDTH,
                                antialiased=ANTIALIAS)
    if VERBOSE:
        show("cost v. parameters surface")
        max_expectation = np.max(expectation)
        print("Max expectation value: %.02f" % max_expectation)


def build_erdos_renyi(num_qubits, edge_probability):
    """
    Builds Erdos-Renyi graph (random graph) representing connectivity.
    """
    # Builds Erdos-Renyi graph (random graph)
    print("NUM QUBITS: ", num_qubits)
    random_graph = nx.erdos_renyi_graph(num_qubits, edge_probability)
    nodes, edges = list(random_graph.nodes), list(random_graph.edges)
    
    # Updates edge weights for random graph
    edge_weights = [1.0] * len(edges)
    for index, edge in enumerate(edges):
        left, right = edge
        random_graph[left][right]['weight'] = edge_weights[index]
    return random_graph


def draw_nx_graph(graph):
    """
    Draw networkx graph with nodes and edges.
    """
    node_colors = ['y'] * len(graph.nodes)
    NODE_SIZE = 400
    TRANSPARENCY = 0
    plt.figure()
    default_axes = plt.axes(frameon=False)
    optimal_node_positioning = nx.spring_layout(graph)
    nx.draw_networkx(graph, node_color=node_colors, node_size=NODE_SIZE, 
                     alpha=1-TRANSPARENCY, ax=default_axes,
                     pos=optimal_node_positioning)


def Ising(circuit, edge, gamma):
    """
    Applies an Ising-type interaction to given edge, connecting qubits.
    """
    left_qubit, right_qubit = edge
    circuit.cp(-2 * gamma, left_qubit, right_qubit) # cp replaces cu1
    circuit.p(gamma, left_qubit)                    # p replaces u1
    circuit.p(gamma, right_qubit)


def basic_qaoa_circuit(nodes, edges, gamma_opt, beta_opt, rounds):
    """
    Builds QAOA circuit using problem ansatz (see arxiv 1411.4028).
    """
    # Builds empty circuit and initializes all qubits into superposition
    num_qubits = len(nodes)
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(nodes)

    for optimization_round in range(rounds):
        # Applies circuit connectivity with Ising type gates
        #circuit.barrier()
        for edge in edges:
            Ising(circuit, edge, gamma_opt)

        # Applies single qubit rotations to generate final state evolution 
        #circuit.barrier()
        circuit.rx(2 * beta_opt, nodes)
    
    # Adds qubit measurements 
    circuit.measure_all()
    return circuit


def ZZ(circuit, edge, gamma):
    """
    Applies a ZZ interaction to given edge, connecting qubits.
    """
    left_qubit, right_qubit = edge
    circuit.cx(left_qubit, right_qubit)
    circuit.rz(-2 * gamma, right_qubit)
    circuit.cx(right_qubit, left_qubit)


def fermionic_swap_circuit(nodes, edges, gamma_opt, beta_opt):
    """
    Builds quantum circuit using optimal parameters, using Fermionic SWAP
     network topology (improving success probability...?).
    """
    
    # Builds empty circuit and initializes all qubits into superposition.
    num_qubits = len(nodes)
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(nodes)

    # Applies circuit connectivity with ZZ interactions, following Fermionic
    #  SWAP network topology. Given N qubits in our circuit, per step we
    #  alternate between swapping the first floor(N/2) pairs and the last
    #  floor(N/2) pairs, stopping after N steps (N * floor(N/2) swaps).
    #circuit.barrier()
    step = 0
    while step < num_qubits:
        pairs = [(idx, idx + 1) for idx in range(0, num_qubits // 2 + 2, 2)]
        for edge in pairs:
            if edge in edges:
                ZZ(circuit, edge, gamma_opt)
        step += 1

        if step == num_qubits:
            break

        pairs = [(idx, idx - 1) for idx in range(num_qubits // 2,
                                                 num_qubits, 2)]
        for edge in pairs:
            if edge in edges:
                ZZ(circuit, edge, gamma_opt)
        step += 1

    # Applies single qubit rotations to generate final state evolution 
    #circuit.barrier()
    circuit.rx(2 * beta_opt, nodes)
    
    # Adds qubit measurements 
    circuit.measure_all()
    return circuit


def build_circuit(nodes, edges, gamma_opt, beta_opt, swap_network=False,
                  rounds=1):
    """
    Builds quantum circuit to prepare trial state, using optimal parameters.
    """
    if swap_network:
        return fermionic_swap_circuit(nodes, edges, gamma_opt, beta_opt)
    return basic_qaoa_circuit(nodes, edges, gamma_opt, beta_opt, rounds)


def save_circuit(circuit, image_filename, pickle=False):
    """
    Saves circuit to a JPEG and a pickle file for debugging.

    Inputs:
     circuit (QuantumCircuit): quantum circuit instance,
     image_filename (str): name to save circuit under.

    No return.
    """
    circuit_image = circuit_drawer(circuit, output="latex") 
    filepath = "images/" + image_filename + ".jpg"
    circuit_image.save(filepath)
    if pickle:
        dump(circuit, open(image_filename + ".p", "wb"))


# Defines cost function to compute cost given bitstring, to determine
# how good candidate bitstring is
def evaluate_cost(graph, bitstr):
    """
    Evaluates canonical cost function for combinatorial
    optimization, corresponding to MAXCUT.
    
    Inputs:
     graph: (networkx graph instance) 
     bitstr: (str) MAXCUT solution bitstring
    
    Returns cost value, else NaN if solution invalid (wrong length).
    """
    
    if len(bitstr) != len(graph.nodes):
        return np.nan
    
    cost = 0
    for edge in graph.edges():
        left_node, right_node = edge
        w = graph[left_node][right_node]['weight']
        cost += w * (int(bitstr[left_node]) * (1 - int(bitstr[right_node]))
                   + int(bitstr[right_node]) * (1 - int(bitstr[left_node])))
    return cost


def evaluate_data(counts, qubit_graph):
    """
    Performs evaluation of generated data, computing predicted bitstring (i.e.,
     that with maximum cost) and expected cost of sampled bitstrings.
    """ 
    # Reports sampled bitstring with maximum cost value
    num_qubits = len(qubit_graph.nodes)
    samples = counts.keys()
    predicted_bitstr = {"bitstr": None, "cost": None}
    expected_cost = 0
    cost_values = {}

    for sample in samples:
        bitstr = sample[:num_qubits] # (?) sampled bitstring 2x qubits long...
        cost = evaluate_cost(qubit_graph, bitstr)
        current_max_cost = predicted_bitstr["cost"]
        if ((predicted_bitstr["bitstr"] == None) or 
            (cost > current_max_cost)):
            predicted_bitstr["bitstr"] = bitstr
            predicted_bitstr["cost"] = cost
    
        # Computes mean energy (?) and check if it agrees with prediction...
        probability = counts[sample] / SHOTS
        expected_cost += probability * cost
    
        # Stores costs in dictionary, for histogram plotting (to confirm
        # that costs concentrate around predicted mean)
        cost_label = str(round(cost))
        cost_values[cost_label] = cost_values.get(cost_label, 0) \
                                + counts[sample]

    return predicted_bitstr, expected_cost, cost_values


def build_backend(error_probability=0):
    """
    Builds a (potentiall noisy) backend.

    Inputs:
     error_probability (float): level of noise to build noise model with

    Returns a quantum backend (i.e., QasmSimulator).
    """
    model = get_noise_model(error_probability)
    backend = (QasmSimulator(noise_model=model)
               if error_probability > 0 else QasmSimulator())
    return backend


def executor(circuit, backend, decomposed=False):
    """
    Given a quantum program, executes it on some backend.
    
    Inputs:
        circuit: quantum circuit implementing some program
        backend: simulator to execute circuit on
        
    Returns sampling counts for circuit execution.
    """

    if VERBOSE:
        print(f"Executing circuit with QASM, {SHOTS} shots...")

    # Only execute circuits decomposed into preset basis.
    BASIS = FOLDING_BASIS_1Q + FOLDING_BASIS_2Q
    if not decomposed: 
        circuit = transpile(circuit, backend, basis_gates=BASIS)
    job = execute(circuit,
                  backend,
                  basis_gates=BASIS,
                  optimization_level=0,
                  shots=SHOTS)
    #job = execute(circuit, backend, shots=SHOTS) #<- basic execution
    counts = job.result().get_counts()
    return counts


def get_noise_model(error_probability):
    """
    Adds depolarizing error channel with given error probability.
    """
    if VERBOSE:
        print("Adding depolarizing error channel with %s error..."
              % error_probability)

    single_qubit_gates = FOLDING_BASIS_1Q
    two_qubit_gates = FOLDING_BASIS_2Q

    noise_model = NoiseModel()
    single_qubit_error = depolarizing_error(error_probability, 1)
    two_qubit_error = depolarizing_error(error_probability, 2)
    noise_model.add_all_qubit_quantum_error(single_qubit_error,
                                            single_qubit_gates)
    noise_model.add_all_qubit_quantum_error(two_qubit_error,
                                            two_qubit_gates)
    return noise_model


def fold_circuit(circuit, scale_factor, backend):
    """
    Applies unitary folding to given circuit, until scale factor is reached. 
    """
    if VERBOSE:
        print("Folding from left with scale factor %s..." % scale_factor)
    
    FOLDING_BASIS = FOLDING_BASIS_1Q + FOLDING_BASIS_2Q 
    decomposed_circuit = transpile(circuit, backend, basis_gates=FOLDING_BASIS)
    save_circuit(decomposed_circuit, "circuit_for_debugging", pickle=True)
    # For now, using fold_gates_from_right -- it returns the same circuit every
    #  time, and we need this to compute the gradients for different parameters
    #  but identical circuits).
    return fold_gates_from_right(decomposed_circuit, scale_factor)


def compute_cost_difference(qubit_graph, parameters, initial_expected_cost,
                            backend,
                            swap_network,
                            fold_scale_factor=0,
                            rounds=1):
    """ 
    Gets partial change in cost from given initial cost, using new parameters.

    Inputs:
     qubit_graph: qubit program interaction graph,
     parameters: gate parameters,
     initial_expected_cost (float):
     swap_network: indicates whether a swap network circuit should be built,
     fold_scale_factor (int):

    Returns difference between newly computed expected cost and initial value.
    """
    nodes, edges = qubit_graph.nodes, qubit_graph.edges
    gamma, beta = parameters
    circuit = build_circuit(nodes, edges, gamma, beta, swap_network, rounds)
    if fold_scale_factor > 1.0:
        circuit = fold_circuit(circuit, fold_scale_factor, backend)
    counts = executor(circuit, backend,
                      decomposed=(fold_scale_factor > 1.0))
    _, expected_cost, _ = evaluate_data(counts, qubit_graph)
    return expected_cost - initial_expected_cost



def compute_gradient_magnitude(qubit_graph, parameters, initial_expected_cost,
                               backend,
                               swap_network,
                               fold_scale_factor=0,
                               rounds=1):
    """
    Computes gradient vector and retrieves magnitude (Euclidean norm).
    """
    gamma, beta = parameters
    PARAMETER_DELTA = 0.1
    gamma_difference = compute_cost_difference(qubit_graph,
                                               [gamma + PARAMETER_DELTA,
                                                beta],
                                               initial_expected_cost,
                                               backend,
                                               swap_network,
                                               fold_scale_factor,
                                               rounds)
    beta_difference = compute_cost_difference(qubit_graph,
                                              [gamma,
                                               beta + PARAMETER_DELTA],
                                              initial_expected_cost,
                                              backend,
                                              swap_network,
                                              fold_scale_factor,
                                              rounds)
    gradient = [gamma_difference / PARAMETER_DELTA,
                beta_difference / PARAMETER_DELTA]
    gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
    return gradient_magnitude


def estimate_gradient(qubit_graph, gamma, beta, error_probability=0,
                      swap_network=False, zne=False,
                      extrapolator=None, rounds=1):
    """
    Estimates gradient magnitudes (and expected costs) for evaluation.
    """
    nodes, edges = qubit_graph.nodes, qubit_graph.edges
    num_qubits = len(nodes)

    # Executes circuit with QASM simulator backend
    circuit = build_circuit(nodes, edges, gamma, beta, swap_network, rounds)
    ############################################################################
    # Saves image of 5-qubit circuit generation, to compare with Pranav/Teague
    #  paper results for completely connected 5-qubit graph...
    if num_qubits == 5 and EDGE_PROBABILITY == 1:
        save_circuit(circuit, "pranav-teague-circuit")
    ###########################################################################
    backend = build_backend(error_probability)
    counts = executor(circuit, backend)
    _, initial_expected_cost, _ = evaluate_data(counts, qubit_graph) 

    gradient_magnitude = compute_gradient_magnitude(qubit_graph,
                                                    [gamma, beta],
                                                    initial_expected_cost,
                                                    backend,
                                                    swap_network,
                                                    rounds)
    if not zne:
        return gradient_magnitude, initial_expected_cost

    # Applies unitary folding to generate gradient magnitudes and subsequently
    #  extrapolates magnitudes to zero-noise case.
    #
    # Note: we can use the same qubit graph to evaluate sampled data, given that
    #  unitary folding only expands the circuit by gates that cancel to the
    #  identity (recall that we only want to see how adding noise in circuit
    #  execution affects the sampling results).
    noise_scalars = [0, 1.0, 2.0, 3.0]
    folding_magnitudes = []
    for scale_factor in noise_scalars:
        if scale_factor == 0:
            ZERO_NOISE_LEVEL = 0
            backend = build_backend(ZERO_NOISE_LEVEL)
            counts = executor(circuit, backend)
        elif scale_factor == 1.0:
            folding_magnitudes.append(gradient_magnitude)
            continue
        else:
            backend = build_backend(error_probability)
            folded_circuit = fold_circuit(circuit, scale_factor, backend)
            counts = executor(folded_circuit, backend, decomposed=True)
        _, init_expec_cost, _ = evaluate_data(counts, qubit_graph)
        folding_gradient_magnitude = compute_gradient_magnitude(qubit_graph,
                                                                [gamma, beta],
                                                                init_expec_cost,
                                                                backend,
                                                                swap_network,
                                                                scale_factor)
        folding_magnitudes.append(folding_gradient_magnitude)

    # Extrapolates computed gradient magnitudes to zero-noise limit (np.polyfit,
    #  under the hood; see https://mitiq.readthedocs.io
    zne_magnitude = extrapolator.extrapolate(noise_scalars[1:],
                                             folding_magnitudes[1:],
                                             order=2)
    return *folding_magnitudes, zne_magnitude


def update_gradient_data(input_size, parameters, noise_level, data_spaces,
                         extrapolator=None,
                         sim_conditions={"nibp": False,
                                         "fermionic": False,
                                         "zne": False,
                                         "rounds": False}):
    """
    Computes expected costs and gradient magnitude data, storing in dictionary
     data structures.

    noise_level: probability of depolarization error
    """

    # Parses simulation conditions, determining which blocks of data to compute.
    nibp_condition = sim_conditions["nibp"]
    fermionic_condition = sim_conditions["fermionic"]
    zne_condition = sim_conditions["zne"]
    rounds_condition = sim_conditions["rounds"]
    if not (nibp_condition or fermionic_condition
         or zne_condition or rounds_condition):
        print("No conditions are set to true -- no computation scheduled.")
        return False 

    gamma, beta = parameters
    if nibp_condition or fermionic_condition or zne_condition:
        num_qubits = input_size
        qubit_graph = build_erdos_renyi(num_qubits, EDGE_PROBABILITY)

    if nibp_condition:
        expected_cost_data, gradient_magnitude_data = data_spaces
        grad_magnitude, expected_cost = estimate_gradient(qubit_graph,
                                                          gamma,
                                                          beta,
                                                          noise_level)
        expected_cost_data["standard"][num_qubits].append(expected_cost)
        gradient_magnitude_data["standard"][num_qubits].append(grad_magnitude)

    if fermionic_condition:
        expected_cost_data, gradient_magnitude_data = data_spaces
        grad_magnitude, expected_cost = estimate_gradient(qubit_graph,
                                                          gamma,
                                                          beta,
                                                          noise_level,
                                                          swap_network=True)
        expected_cost_data["fermionic"][num_qubits].append(expected_cost)
        gradient_magnitude_data["fermionic"][num_qubits].append(grad_magnitude)

    if zne_condition:
        expected_cost_data, gradient_magnitude_data = data_spaces
        gradient_mags = estimate_gradient(qubit_graph,
                                          gamma,
                                          beta,
                                          noise_level,
                                          zne=True,
                                          extrapolator=extrapolator)
        noise_free, no_fold, double_fold, triple_fold, zne_mag = gradient_mags
        gradient_magnitude_data["standard"][num_qubits].append(noise_free)
        gradient_magnitude_data["no fold"][num_qubits].append(no_fold)
        gradient_magnitude_data["double fold"][num_qubits].append(double_fold)
        gradient_magnitude_data["triple fold"][num_qubits].append(triple_fold)
        gradient_magnitude_data["zne"][num_qubits].append(zne_mag)

    if rounds_condition:
        num_rounds = input_size
        rounds_data, qubit_graph = data_spaces
        gradient_mags = estimate_gradient(qubit_graph,
                                          gamma,
                                          beta,
                                          noise_level,
                                          #zne=True,
                                          extrapolator=extrapolator,
                                          rounds=num_rounds)
        gradient_magnitude, expected_cost = gradient_mags
        rounds_data["standard-rounds"][num_rounds].append(gradient_magnitude)
        rounds_data["variances"][num_rounds].append(expected_cost)
        #rounds_data["zne-rounds"][num_rounds].append(zne_mag)

    return True


def get_data_space(data_key, simulation_conditions, input_range):
    """
    Gets space of data to later fill with computed data, depending on simulation
     conditions.

    Inputs:
     data_key: a key to the dataset dictionary,
     simulation_conditions: conditions determining which data to compute,
     input_range: range of input over which to define empty data dictionary,
                  mapping inputs to an empty list of outputs.

    Returns a dictionary mapping integers to empty lists, or None.
    """
    
    # Parses simulation conditions, determining which blocks of data to compute.
    nibp_condition = simulation_conditions["nibp"]
    fermionic_condition = simulation_conditions["fermionic"]
    zne_condition = simulation_conditions["zne"]
    rounds_condition = simulation_conditions["rounds"]
    if not (nibp_condition
         or fermionic_condition
         or zne_condition
         or rounds_condition):
        return None

    NIBP_KEYS = {"standard"}
    FERMIONIC_KEYS = {"fermionic"}
    ZNE_KEYS = {"standard", "no fold", "double fold", "triple fold", "zne"}
    ROUNDS_KEYS = {"standard-rounds", "variances", "zne-rounds"}

    if ((data_key in NIBP_KEYS and nibp_condition) or
        (data_key in FERMIONIC_KEYS and fermionic_condition) or
        (data_key in ZNE_KEYS and zne_condition) or
        (data_key in ROUNDS_KEYS and rounds_condition)):
        data_dict = {step: [] for step in input_range}
        return data_dict


def simulate(input_ranges, num_trials, error_probability, sim_conditions,
             fixed_qubit_graph):
    """
    Simulates circuit and evaluates data for some number of trials, returning
     expected costs and gradient magnitudes.
    """
    # Over some number of trials, simulates quantum circuit(s), possibly in the
    #  presence of noise, and computes gradient and expected cost values for
    #  later plotting.
    num_qubits_range, num_rounds_range = input_ranges
    expected_cost_keys = ["standard", "fermionic"]
    nibps_keys = ["standard", "fermionic", "no fold",
                  "double fold", "triple fold", "zne"]
    rounds_keys = ["standard-rounds", "variances", "zne-rounds"] 
    expected_cost_data = {key: get_data_space(key, sim_conditions,
                                              num_qubits_range)
                          for key in expected_cost_keys}
    gradient_magnitude_data = {key: get_data_space(key, sim_conditions,
                                                   num_qubits_range)
                               for key in nibps_keys}
    rounds_data = {key: get_data_space(key, sim_conditions, num_rounds_range)
                   for key in rounds_keys}

    poly_factory = PolyFactory(scale_factors=[1.0, 2.0, 3.0], order=2)
    for trial in range(num_trials):
        gamma, beta = random() * pi, random() * pi
        new_conditions = {key: value for key, value in sim_conditions.items()}
        new_conditions["rounds"] = False
        for num_qubits in num_qubits_range:
            parameters = (gamma, beta)
            data_spaces = (expected_cost_data, gradient_magnitude_data)
            executed = update_gradient_data(num_qubits, parameters,
                                            error_probability,
                                            data_spaces,
                                            extrapolator=poly_factory,
                                            sim_conditions=new_conditions)
            if not executed:
                print("No data was computed (check conditions).")
    
        for num_rounds in num_rounds_range:
            parameters = (gamma, beta)
            data_spaces = rounds_data, fixed_qubit_graph
            executed = update_gradient_data(num_rounds, parameters,
                                            error_probability,
                                            data_spaces,
                                            extrapolator=poly_factory,
                                            sim_conditions=sim_conditions)
            if not executed:
                print("No data was computed (check conditions).")

    print('=', end='', flush=True) # a simple loading bar...
    return expected_cost_data, gradient_magnitude_data, rounds_data


#------------------------------------------------------------------------------#
def basic_simulate(nodes, edges, gamma_opt, beta_opt, noisy=False,
                   swap_network=False):
    """
    Runs simulation on circuit given by set of qubits (nodes) and connectivity
     (edges), using given optimal parameters. 
    """
    num_qubits = len(nodes)

    # Builds quantum circuit to prepare trial state, using optimal parameters
    circuit = build_circuit(nodes, edges, gamma_opt, beta_opt, swap_network)
    filename = "noisy_basic_circuit" if noisy else "basic_circuit"
    save_circuit(circuit, filename)
    
    # Executes circuit with QASM simulator backend and plots results
    if not noisy:
        counts = executor(circuit)
        if PLOT:
            plot_histogram(counts, bar_labels=False)
            show("noiseless bitstring probabilities")
    else:
        # Adds depolarizing error channel with probabiltiy 0.001
        error_probability = 0.001
        counts = executor(circuit, error_probability=error_probability)

    # Evaluates data...
    predicted_bitstr, expected_cost, cost_values = evaluate_data(counts,
                                                                 qubit_graph)
    if VERBOSE:
        print("Predicted bitstring solution: %s" % predicted_bitstr["bitstr"])
        print("Cost: %f" % predicted_bitstr["cost"])
        print("Expected cost (energy): %s" % expected_cost)
    
    # Plots cost function distribution (over bitstrings sampled)
    if PLOT:
        plot_histogram(cost_values)
        show("noiseless cost distribution")
        if noisy:
            # Plots cost function distribution (over bitstrings sampled)
            show("noisy cost distirbution")
            plot_histogram(cost_values)

    return predicted_bitstr, expected_cost
#------------------------------------------------------------------------------#

if __name__ == "__main__":
    pass
