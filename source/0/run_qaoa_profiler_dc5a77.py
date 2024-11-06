# https://github.com/lfd/arcs2022/blob/f03319dab587c1543e46fc929b3d05d4c5bbaca0/run_qaoa_profiler.py
# Copyright (c) Siemens AG, 2022
# SPDX-License-Identifier: GPL-2.0


import numpy as np
from scipy.optimize import minimize
import itertools
import networkx as nx

from qiskit.test.mock import FakeBrooklyn
from qiskit import Aer, QuantumCircuit,  transpile, execute
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import SLSQP, COBYLA

from qiskit.circuit import Parameter


@profile
def generate_graph_from_density(num_nodes, density):

    """
    Input: 
    num_nodes: number of nodes  
    density: coupling density = number of edges / maximum number of edges

    Output:
    Graph G
    """

    G = nx.Graph()
    nodes=np.linspace(0, num_nodes-1, num_nodes).astype(int)
    G.add_nodes_from(nodes)

    #Randomly set edges with the given density
    max_edge_list = list(itertools.combinations(nodes, 2)) # all egdes that would be possible
    max_edges = len(max_edge_list)
    num_edges = np.round(max_edges*density).astype(int)

    edges=[]
    for j in range(num_edges):
        edge=max_edge_list[np.random.choice(np.arange(len(max_edge_list)), size=1, replace=False)[0]]
        edges.append(edge)
        max_edge_list.remove(edge)

    G.add_edges_from(edges)

    return G



@profile
def get_qubo_maxcut(G):
    """
    Parameters:
    G: Graph with nodes and edges

    Output: QUBO-matrix
    """

    n_qubits=len(list(G.nodes))

    H=np.zeros((n_qubits, n_qubits))
    for pair in list(G.edges()):
        H[pair[0],pair[0]]+=-1
        H[pair[1],pair[1]]+=-1
        H[pair[0],pair[1]]+=2
    return H


@profile
def get_simple_circ_qaoa_from_qubo(Q):

    """
    Outputs parametrized circuit for running QAOA with single layer.
    """
    
    gamma = Parameter('gamma')
    beta =  Parameter('beta')
    # get number of qubits
    num_qubits = len(Q) 

    # index tuples for 2-qubit gates: All possible combinations [i,j] for i!=j and i,j \in {1,num_qubits} whereas [i,j] = [j,i]
    tq_id = list(itertools.combinations(np.linspace(0, num_qubits-1, num_qubits, dtype=int), 2))
    
    Ising_h = []
    for i in range(num_qubits):
        Ising_h.append(0.5*Q[i,i] + 0.25*np.sum([(Q[i,j]+Q[j,i]) for j in range(num_qubits) if j != i]))
     
    
    # quadratic Ising terms: Use J[i,j] only for unequal indices [[i,j]
    Ising_J = 0.25*Q

    sq_gates={}
    tq_gates={}
    for j in range(num_qubits):
        sq_gates[j] = j, Ising_h[j]
    for j in range(len(tq_id)):
        tq_gates[j] = tq_id[j], Ising_J[tq_id[j][0], tq_id[j][1]] 

  
    qc = QuantumCircuit(num_qubits)
    
    # Intial state: Eigenstate of mixing Hamiltonian
    for i in range(0, num_qubits):
        qc.h(i)
        

    # Problem Hamiltonian 
    for key in tq_gates.keys():  # two-qubit gates
        qc.rzz(2 * gamma * tq_gates[key][1], tq_gates[key][0][0], tq_gates[key][0][1])
        
    for key in sq_gates.keys():  # single-qubit gates
        qc.rz(2 * gamma * sq_gates[key][1] , sq_gates[key][0])

    # Mixing Hamiltonian
    for i in range(0, num_qubits):
        qc.rx(2 * beta, i)
            
    qc.measure_all()
        
    return qc

@profile
def execute_circ(circ, theta, backend, shots):

    gamma_val, beta_val = theta
    
    qc = circ.assign_parameters({circ.parameters[0]: gamma_val, circ.parameters[1]: beta_val})

    job_sim = backend.run(qc, seed_simulator=123, nshots=shots)
    counts = job_sim.result().get_counts()
    return counts

@profile
def compute_expectation_val(H, counts):
    avg=0
    sum_counts = 0
    for bitstring, count in counts.items():
        vec = np.array([*map(int, bitstring)]) 
        avg += np.dot(vec, np.dot(H, vec))*count
        sum_counts += count

    return avg/sum_counts

@profile
def get_expectation(theta, args):

    circ, backend, shots = args   
    counts=execute_circ(circ, theta, backend, shots)

    return compute_expectation_val(H, counts) 

@profile
def minimize_qaoa(initial, qc_compiled, backend, shots):
    res = minimize(get_expectation, initial, args=[qc_compiled, backend, shots], method='COBYLA')
    return res



backend_gates = FakeBrooklyn()
config = backend_gates.configuration()
gate_set=config.basis_gates

num_nodes = 20
density = 0.5
p=1
initial=[0.1, 0.1]
opt_level=3

seed=123
shots=10000

backend=Aer.get_backend("qasm_simulator")
algorithm_globals.random_seed = seed
optimizer=SLSQP(maxiter=1000)
params=np.repeat(initial, p)

G = generate_graph_from_density(num_nodes, density)
H = get_qubo_maxcut(G)
    
qc = get_simple_circ_qaoa_from_qubo(H)
qc_compiled = transpile(qc, basis_gates=gate_set, optimization_level=opt_level)


res = minimize_qaoa(initial, qc_compiled, backend, shots)