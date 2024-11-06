# https://github.com/ampolloreno/qaoa/blob/519fff304142e5b9bc1d34e8dc1c4b2e8ac5ea37/classical_optimization/classical_optimization/DA_cluster.py
import quimb as qu
import sys
import quimb.tensor as qtn
import cotengra as ctg
from scipy.optimize import dual_annealing
from classical_optimization.qaoa_circuits import execute_qaoa_circuit_and_estimate_cost
import numpy as np
from qiskit import Aer, execute
from coldquanta.qiskit_tools.modeling.neutral_atom_noise_model import create_noise_model
import networkx as nx

shots_per_point = sys.argv[1]
maxiter = sys.argv[2]
initial_temp = sys.argv[3]
restart_temp_ratio = sys.arg[4]

opt = ctg.ReusableHyperOptimizer(
    reconf_opts={},
    max_repeats=16,
    parallel=True,
)


np.random.seed(666)
reprate = 50
one_hour = 60 * 60  # seconds
max_gamma = 2 * np.pi
max_beta = np.pi
simulator = Aer.get_backend('qasm_simulator')
noise_model = create_noise_model(cz_fidelity=1)

reg = 3
n = 20
seed = 666
np.random.seed(seed)
graph = nx.watts_strogatz_graph(n, reg, .5, seed=seed)
terms = {(i, j): np.random.rand() for i, j in graph.edges}
circ_ex = None


def weights(graph):
    rtn = {}
    for e in graph.edges:
        weight = graph.get_edge_data(e[0], e[1])['weight']
        rtn[e] = weight
    return rtn


def objective(terms):
    history = []

    def store_log(func):
        def logged_func(x):
            ret = func(x)
            history.append((x, ret))
            return ret

        return logged_func

    @store_log
    def gamma_beta_objective(gamma_beta):
        p = 1
        gammas = [gamma_beta[1]]
        betas = [gamma_beta[0]]
        circ_ex = qtn.circuit_gen.circ_qaoa(terms, p, gammas, betas)
        ZZ = qu.pauli('Z') & qu.pauli('Z')
        ens = [
            circ_ex.local_expectation(weight * ZZ, edge, optimize=opt)
            for edge, weight in terms.items()
        ]

        return sum(ens).real
    return gamma_beta_objective, history


func, history2 = objective(terms)
initial_gamma_beta = [np.random.rand() * max_param for max_param in (max_gamma, max_beta)]
result = dual_annealing(
    lambda x: -1 * func(x),
    bounds=[(0, max_gamma),
            (0, max_beta)],
    x0=np.array(initial_gamma_beta),
    # One annealing attempt.
    maxiter=maxiter,
    initial_temp=initial_temp,
    maxfun=one_hour * reprate,
    restart_temp_ratio=restart_temp_ratio,
    no_local_search=True,
    seed=seed)
result.fun = -result.fun


def write_data():
    h = hashlib.md5()
    arr = np.array(list(result.x) +  [result.fun] + history)
    h.update(arr)
    name = 'DA'
    shots_per_point = sys.argv[1]
    maxiter = sys.argv[2]
    initial_temp = sys.argv[3]
    restart_temp_ratio = sys.arg[4]
    filename = f'{name}/{shots_per_point}_{maxiter}_{initial_temp}_{restart_temp_ratio}.pkl'
    try:
        os.mkdir(f'{name}')
    except FileExistsError:
        pass
    try:
        with open(filename, 'rb') as filehandle:
            data = dill.load(filehandle)
            print("Fetching existing file...")
    except FileNotFoundError:
        data = {'data': arr}
    with open(filename, 'wb') as filehandle:
        dill.dump(data, filehandle)
    return filename

write_data()