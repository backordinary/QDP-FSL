# https://github.com/Mephphisto/QAOA-Demo/blob/c0e9ed84b766dd17e3ac8fc99782fb040cb33774/QAOA_MAX_cut.py
import qiskit as qs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict
from operator import itemgetter
from numba import jit


def hamiltonian_trotter_term(Graph, gamma):
    '''
    This produces a Trotter term of the Ising Hamiltonian for Graph
    :param Graph: Graph
    :param gamma: gamma parameter
    :return: Quantum Gate representing Trotter term of Hamiltonoan Exponential
    '''
    N = Graph.number_of_nodes()
    qc = qs.QuantumCircuit(N, N)
    for j, k in Graph.edges():
        qc = qc.compose(pauli_zz_gate(N, j, k, gamma))
    return qc


def pauli_zz_gate(N, j, k, gamma):
    '''
    Produces a Pauli $Z_i \otimes Z_j$ gate
    :param N: number uf Qubits
    :param j: index j
    :param k: index k
    :param gamma: gamma parameter
    :return: quantum ZZ Gate
    '''
    qc = qs.QuantumCircuit(N, N)
    qc.cx(j, k)
    qc.rx(2 * gamma, k)
    qc.cx(j, k)
    return qc


def QAOA_Mixer(Graph, beta):
    '''
    QAOA Mixer Gates rotate each qubit by $\beta$ around x-Axis
    :param Graph: Graph
    :param beta: beta parameter
    :return: quantum Mixer Circuit
    '''
    N = Graph.number_of_nodes()
    qc = qs.QuantumCircuit(N, N)
    for j in Graph.nodes():
        qc.rx(2 * beta, j)
    return qc


def QAOA_Cirquit(Graph, p, beta, gamma):
    '''
     Generator for QAOA circuit from a Graph object, $\beta$ and $\gamma$ arrays
    :param Graph: Graph
    :param p: number of  trotter terms
    :param beta: beta parameters
    :param gamma: gamma parameters
    :return: Quantum circuit
    '''
    # Ensure correct array lengths
    assert (len(beta) == len(gamma) == p)
    N = Graph.number_of_nodes();
    qc = qs.QuantumCircuit(N, N)
    # Initialize superposition
    qc.h(range(N))
    # Trotter Product of $exp(- \beta B + i \gamma C)$ with p terms
    for k in range(p):
        qc = qc.compose(hamiltonian_trotter_term(Graph, gamma[k]))
        qc = qc.compose(QAOA_Mixer(Graph, beta[k]))
    # barrier and measurement
    qc.barrier(range(N))
    qc.measure(range(N), range(N))
    return qc


def QAOA_Error(index, Graph):
    '''
    QAOA Optimizer Error function
    :param index: index
    :param Graph: Graph
    :return: Edges cut by Max cut
    '''
    res = 0
    for j, k in Graph.edges():
        if index[j] == index[k]: res += 1

    return res


def Ising_Energy(counts, Graph):
    '''
    Computes the Expectaiton value of the Ising Hamiltonian for Graph G
    :param counts: counted output from quantum computation
    :param Graph: Graph
    :return: <E>
    '''
    E, Num = 0, 0
    for idx, cnt in counts.items():
        E += QAOA_Error(idx, Graph) * cnt
        Num += cnt

    return float(E / Num)


def Func_Gen(Graph, p, backend):
    '''
    Generator for functors needed to do optimisation with scipy
    :param Graph: Graph
    :param p: number of  trotter terms
    :param backend: backend to run quantum computation on
    :return:
    '''

    @jit(forceobj=True)
    def f(theta):
        # let's assume first half is betas, second half is gammas
        beta = theta[:p]
        gamma = theta[p:]
        qc = QAOA_Cirquit(Graph, p, beta, gamma)
        counts = qs.execute(qc, backend).result().get_counts()
        # return the energy
        return -1 * Ising_Energy(counts, Graph)

    return f


def Optimize_Params(Graph, p, backend):
    '''
    Optimize parameters
    :param Graph: Graph
    :param p: number of  trotter terms
    :param backend: backend to run quantum computation on
    :return: optimal parameters
    '''
    init_params = np.ones(2 * p)
    func = Func_Gen(Graph, p, backend)
    return scipy.optimize.minimize(func, init_params, method='COBYLA', options={'maxiter': 1000})


def MAX_Cut(Graph, backend):
    '''
    This is the implementation of max Cut bassed on QAOA
    :param Graph: Grap to be cur
    :param backend: Quantum computing backend of choice
    :return: best_cut, best_solution
    '''
    qc = hamiltonian_trotter_term(Graph, np.pi)
    job = qs.execute(qc, backend)
    result = job.result()
    result.get_counts()
    params = Optimize_Params(Graph, 5, backend)['x']
    qc = QAOA_Cirquit(Graph, 5, params[:5], params[5:])
    counts = qs.execute(qc, backend).result().get_counts()
    # get the best solution:
    return min([(QAOA_Error(x, Graph), x) for x in counts.keys()], key=itemgetter(0))


def QAOA_Test():
    '''
    This function is used to test the implementation
    :return:
    '''
    Graph = nx.erdos_renyi_graph(5, 0.75)
    qc = hamiltonian_trotter_term(Graph, np.pi)
    backend = qs.Aer.get_backend('qasm_simulator')
    job = qs.execute(qc, backend)
    result = job.result()
    result.get_counts()
    params = Optimize_Params(Graph, 5, backend)['x']
    qc = QAOA_Cirquit(Graph, 5, params[:5], params[5:])
    counts = qs.execute(qc, backend).result().get_counts()
    # get the best solution:
    best_cut, best_solution = min([(QAOA_Error(x, Graph), x) for x in counts.keys()], key=itemgetter(0))
    print(f"Best string: {best_solution} with cut: {best_cut}")
