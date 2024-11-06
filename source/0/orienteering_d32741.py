# https://github.com/StevenSchuerstedt/QuantumComputing/blob/a467353b2e896fc10b87c10b5dded5005c77f436/code/orienteering.py
import networkx as nx
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from qiskit import Aer
from qiskit_optimization.translators import from_docplex_mp
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms import NumPyMinimumEigensolver, VQE, QAOA
from typing import Dict, List
from random import randrange
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# IBMQ.save_account('your api key')
# before using load_account() you need to call save account one time

IBMQ.load_account()

#ibm-q is the public provider
provider = IBMQ.get_provider(hub = 'ibm-q')

# filter for specific backends which are no simulator
small_devices = provider.backends(filters=lambda x: not x.configuration().simulator)

# helper function to get the least busy device
least_busy(small_devices)

### some options
showGraph = True
solveClassically = True
solveQuantumlike = False
solveWithVQEb = True
solveWithQAOAb = False
printConstraints = False


def drawGraph(G):
    reducedEdges = list(filter(lambda x: x[2]['weight'] < G.graph['maxCost'], G.edges(data=True)))
    reducedGraph = nx.DiGraph()
    reducedGraph.add_edges_from(reducedEdges)
    reducedGraph.add_nodes_from(G.nodes)
    labels = {n: str(n) + ': ' + str(G.nodes[n]['weight']) for n in G.nodes}
    colors = [G.nodes[n]['weight'] for n in G.nodes]
    pos = nx.spring_layout(reducedGraph)
    nx.draw(reducedGraph, with_labels=True, labels=labels, node_color=colors, node_size=1000)
    edge_labels = nx.get_edge_attributes(reducedGraph, 'weight')
    nx.draw_networkx_edge_labels(reducedGraph, pos=pos, edge_labels=edge_labels)
    plt.show()

def to_integer_program(G, solveClassically):
    mdl = Model(name="Orienteering")
    n = G.number_of_nodes()
    x = {
        (i, j): mdl.binary_var(name="x_{0}_{1}".format(i, j))
        for (i, j) in G.edges
    }

    u = {
        i : mdl.integer_var(name="u_" + str(i), lb=2, ub=n)
        for i in range(1, n)
    }

    orienteering_func = mdl.sum(
        G.nodes[j]["weight"] * x[(i, j)]
        for (i, j) in G.edges
    )
    mdl.maximize(orienteering_func)

    for i in range(1, n):
        for j in range(1, n):
            if (i, j) in G.edges:
                mdl.add_constraint(u[i] - u[j] + 1 <= (n - 1)*(1 - x[(i, j)]))
            else:
                mdl.add_constraint(u[i] - u[j] + 1 <= (n - 1))

    for k in range(1, n):
        # a node has to have same number of in and out going edges
        # => there are no dead ends
        mdl.add_constraint(mdl.sum(x[i, k] for i in range(n) if (i, k) in G.edges) == mdl.sum(x[k, j] for j in range(n) if (k, j) in G.edges))
        # a node may appear only once 
        mdl.add_constraint(mdl.sum(x[k, j] for j in range(n) if (k, j) in G.edges) <= 1)

    
    # we start and end at node 0
    # at least together with the above constraints this is what this is for
    mdl.add_constraint(mdl.sum(x[0, j] for j in range(1, n) if (0, j) in G.edges) == 1)
    mdl.add_constraint(mdl.sum(x[j, 0] for j in range(1, n) if (j, 0) in G.edges) == 1)

    # do not exceed threshold
    mdl.add_constraint(mdl.sum(
                G.edges[i, j]["weight"] * x[(i, j)]
                for (i, j) in G.edges
            ) <= G.graph['maxCost'])

    if printConstraints:
        mdl.prettyprint()
    if solveClassically:
        print("Classical solution:")
        mdl.solve()
        if mdl.get_solve_status().value == 3:
            print("No solution found.")
        else:
            print("Best route cost: " + str(mdl.blended_objective_values[0]))
            solution = []
            for i in range(n):
                for j in range(n):
                    var = mdl.get_var_by_name(x[i, j].lp_name).raw_solution_value
                    if var > 0:
                        solution.append(str(x[i, j]))

            print("Best route: " + str(beautifySolution(solution)))
    op = from_docplex_mp(mdl)
    return op

def beautifySolution(varList):
    tups = [x.split('_')[1:] for x in varList]
    res = []
    res.append(tups[0][0])
    current = tups[0][1]
    while current != '0':
        el = list(filter(lambda x: x[0] == current, tups))[0]
        res.append(el[0])
        current = el[1]
    return res

        

def interpret(res, qp):
    bits = []
    for i, c in enumerate(bin(res)[:1:-1], 1):
        if c == '1':
            bits.append(i - 1)
    
    vars = [qp.variables[i].name for i in range(len(qp.variables)) if i in bits]
    vars = list(filter(lambda x: x[0] != 'u', vars))
    vars.sort()
    return beautifySolution(vars)


def getExampleGraph1():
    # create a simple orienteering graph (directed)
    G = nx.DiGraph(maxCost=20)
    G.add_node(0, weight=2)
    G.add_node(1, weight=8)
    G.add_node(2, weight=6)
    G.add_node(3, weight=2)
    G.add_edge(0, 1, weight=5 )
    G.add_edge(1, 0, weight=5 )
    G.add_edge(2, 3, weight=5)
    G.add_edge(3, 2, weight=5)
    return G

def getExampleGraph():
    # create a simple orienteering graph (directed)
    G = nx.DiGraph(maxCost=20)
    G.add_node(0, weight=2)
    G.add_node(1, weight=8)
    G.add_node(2, weight=6)
    G.add_node(3, weight=2)
    G.add_edge(0, 1, weight=10 )
    G.add_edge(1, 0, weight=20 )
    G.add_edge(1, 2, weight=5)
    G.add_edge(2, 3, weight=5)
    G.add_edge(3, 0, weight=5)
    G.add_edge(2, 0, weight=5)
    G.add_edge(0, 2, weight=10)
    return G


def getSmallExmapleGraph():
    G = nx.DiGraph(maxCost=20)
    G.add_node(0, weight=2)
    G.add_node(1, weight=8)
    G.add_node(2, weight=6)
    G.add_edge(0, 1, weight=10 )
    G.add_edge(1, 0, weight=20 )
    G.add_edge(1, 2, weight=5)
    G.add_edge(2, 0, weight=5)
    return G


def solveWithQC(qp):
    #convert integer program to qubo
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)

    #convert to ising hamiltonian
    qubitOp, offset = qubo.to_ising()

    #solve using numpy
    ee = NumPyMinimumEigensolver()
    result = ee.compute_minimum_eigenvalue(qubitOp)

    #print('energy:', result.eigenvalue.real)
    print("Quantum solution:")
    print('Best route cost:', (result.eigenvalue.real + offset) * -1)
    x = np.argmax(result.eigenstate.primitive)
    interpretation = interpret(x, qp)
    print('Best route:', interpretation)

def solveWithVQE(qp):
    # convert integer program to qubo
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)

    # convert to ising hamiltonian
    qubitOp, offset = qubo.to_ising()

    algorithm_globals.random_seed = 123
    seed = 10598
    # only statevector simulator works for small instances?
    # use qasm_simulator as backend for larger instances
    #backend = Aer.get_backend('aer_simulator_statevector')

    # get real quantum backend
    backend = least_busy(small_devices)

    quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    spsa = SPSA(maxiter=300)
    ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=5, entanglement='linear')
    print("Solve with VQE...")
    print("number of qubits: ", qubitOp.num_qubits)
    vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)

    result = vqe.compute_minimum_eigenvalue(qubitOp)

    print("VQE solution:")
    print('Best route cost:', (result.eigenvalue.real + offset) * -1)
    x = np.argmax(result.eigenstate.primitive)
    interpretation = interpret(x, qp)
    print('Best route:', interpretation)
    print('time:', result.optimizer_time)


def solveWithQAOA(qp):
    # convert integer program to qubo
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)

    # convert to ising hamiltonian
    qubitOp, offset = qubo.to_ising()

    algorithm_globals.random_seed = 10598
    print("Solve with QAOA...")
    print("number of qubits: ", qubitOp.num_qubits)
    optimizer = COBYLA()
    qaoa = QAOA(qubitOp, optimizer, quantum_instance=Aer.get_backend('statevector_simulator'))

    result = qaoa.compute_minimum_eigenvalue()

    print("QAOA solution:")
    print('Best route cost:', (result.eigenvalue.real + offset) * -1)
    x = np.argmax(result.eigenstate.primitive)
    interpretation = interpret(x, qp)
    print('Best route:', interpretation)
    print('time:', result.optimizer_time)


def makeFullyConnectedGraph(G):
    for (i, j) in [(i, j) for i in range(len(G.nodes)) for j in range(len(G.nodes))]:
        if (i, j) not in G.edges:
            G.add_edge(i, j, weight=G.graph['maxCost'] * 2)


def getRandomGraph(numNodes=10, numEdges=20):
    G = nx.DiGraph(maxCost=100)
    for i in range(numNodes):
        G.add_node(i, weight=randrange(10))

    for i in range(numEdges):
        G.add_edge(randrange(numNodes), randrange(numNodes), weight=randrange(10))

    return G

G = getSmallExmapleGraph()
#G = getExampleGraph()
#G = getRandomGraph(10, 20)
makeFullyConnectedGraph(G)

if showGraph:
    drawGraph(G)
    
#encode problem as constraint integer program
qp = to_integer_program(G, solveClassically)

if solveQuantumlike:
    solveWithQC(qp)

if solveWithVQEb:
    solveWithVQE(qp)

if solveWithQAOAb:
    solveWithQAOA(qp)


