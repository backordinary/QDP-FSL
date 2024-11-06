# https://github.com/milanleonard/SCNC3102/blob/4e935f7b4afcff0732d2b6533c283b444c464179/Code/meshgridnoise.py
#%% Imports
from pennylane import numpy as pnp # get pennylane's numpy wrapper
import numpy as np
import pennylane as qml
from itertools import combinations, groupby
import qiskit
import random
import networkx as nx
from pennylane import expval, var
from functools import partial
from collections import defaultdict
import qiskit.providers.aer.noise as noise
pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z)

grid_size = 50
def gnp_random_connected_graph(n, p, seed):
    """Generate a random connected graph
    n     : int, number of nodes
    p     : float in [0,1]. Probability of creating an edge
    seed  : int for initialising randomness
    """
    random.seed(seed)
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z)

def qaoa_maxcut_grid_noise(graph, n_layers, shots=5000, NoiseModel=None):

    n_wires = len(graph.nodes)
    edges = graph.edges

    def U_B(beta):
        for wire in range(n_wires):
            qml.RX(2 * beta, wires=wire)

    def U_C(gamma):
        for edge in edges:
            wire1 = edge[0]
            wire2 = edge[1]
            qml.CNOT(wires=[wire1, wire2])
            qml.RZ(gamma, wires=wire2)
            qml.CNOT(wires=[wire1, wire2])
    if NoiseModel:
        dev = qml.device("qiskit.aer", wires=n_wires, shots=shots, noise_model=NoiseModel)
    else:
        dev = qml.device("default.qubit", wires=n_wires, shots=shots, analytic=False)

    @qml.qnode(dev)
    def circuit(gammas, betas, edge=None, n_layers=1, n_wires=1):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        for i,j in zip(range(n_wires),range(n_layers)):
            U_C(gammas[i,j])
            U_B(betas[i,j])
        
        return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))
    np.random.seed(42)
    init_params = 0.01 * np.random.rand(2, n_wires, n_layers)
    
    def obj_wrapper(params):
        objstart = partial(objective, params, True, False)
        objend = partial(objective, params, False, True)
        return np.vectorize(objstart), np.vectorize(objend)
    
    def objective(params, start=False, end=False, X=None, Y=None):
        gammas = params[0]
        betas = params[1]
        if start:
            gammas[0,0] = X
            betas[0,0] = Y
        elif end:
            gammas[-1,0] = X
            betas[-1,0] = Y 
        neg_obj = 0
        for edge in edges:
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers, n_wires=n_wires))
        return neg_obj

    X, Y = np.meshgrid(np.linspace(-np.pi,np.pi,grid_size),np.linspace(-np.pi,np.pi,grid_size))
    objstart, objend = obj_wrapper(init_params)
    meshgridfirststartparams = objstart(X, Y)
    meshgridfirstlastparams = objend(X,Y)

    return meshgridfirststartparams#, meshgridfirstlastparams


if __name__ == "__main__":
    import multiprocessing
    Noise_Modelling = False
    Shots_Modelling = True
    TEST_G = gnp_random_connected_graph(6,0.2,42)
    if Noise_Modelling:
        noise_args = np.linspace(0,0.15,15)
        Noise_Models = [noise.NoiseModel() for i in range(10)]
        for noise_arg, noisemodel in zip(noise_args, Noise_Models):
            noisemodel.add_all_qubit_quantum_error(noise.depolarizing_error(noise_arg ,1), ['u1','u2','u3'])
        args = [(TEST_G, 3, 5000, noisemodel) for noisemodel in Noise_Models]
        OUTPUT_ARR = np.zeros((len(Noise_Models),grid_size,grid_size))
    elif Shots_Modelling:
        shots_arr = np.arange(5,101,1)
        args = [(TEST_G,4,i,None) for i in shots_arr]
        OUTPUT_ARR = np.zeros((len(shots_arr),grid_size,grid_size)) 
    
    with multiprocessing.Pool(multiprocessing.cpu_count()-2) as p:
        results = p.starmap(qaoa_maxcut_grid_noise, args)
    for idx, result in enumerate(results):
        OUTPUT_ARR[idx] = result

    np.save("./datafiles/meshgridsnoisy.npy", OUTPUT_ARR) if Noise_Modelling else np.save("./datafiles/meshgridshots.npy",OUTPUT_ARR)
