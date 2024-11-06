# https://github.com/incud/GraphEditDistance/blob/25fc7626cc29582f2a661e90e8695e3bf0c6c0c5/qaoasolutioncreator.py
import warnings
warnings.filterwarnings("ignore")

from os import path
import pandas as pd
from experimentcreator import generate_hamiltonian

from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization import QuadraticProgram
from qiskit.algorithms.optimizers.cobyla import COBYLA
from qiskit.optimization.applications.ising.common import sample_most_likely


def generate_qaoa_dataframe(the_path):
    if path.exists(the_path):
        print("generate_qaoa_dataframe: Loaded from file")
        return pd.read_pickle(the_path)
    else:
        columns = ["experiment", "start", "end",
                   "best_sample", "best_energy", "best_energy_by_sample",
                   "p", "max_iter"]
        return pd.DataFrame(columns=columns)


def replace_variable_name(name, map_names):
    new_name = name.replace("][", "_").replace("[", "_").replace("]", "")
    map_names[name] = new_name
    return new_name


def vector_to_sample(vector, map_names, variables):
    inv_map = {v: k for k, v in map_names.items()}
    sample = {}
    for i in range(len(vector)):
        var = inv_map[variables[i]]
        sample[var] = int(vector[i])
    return sample


def run_qaoa_experiment(experiment, queue, p, max_iter):

    bqm = experiment["bqm"]
    hamiltonian = generate_hamiltonian(experiment["g1"], experiment["g2"], experiment["a"], experiment["b"])

    # ===========================================================
    # CREATE OPERATOR
    model = hamiltonian.compile()
    qubo, offset = model.to_qubo()
    # rename variables from x[0] to x_0
    map_names = {}
    qubo = {(replace_variable_name(x1, map_names), replace_variable_name(x2, map_names)): v
            for (x1, x2), v in qubo.items()}
    variables = sorted(map_names.values())
    # create DOcplex quadratic program
    cplex_program = QuadraticProgram()
    # add the variables
    for x in variables:
        cplex_program.binary_var(x)
    # add the QUBO problem to DOcplex
    cplex_program.minimize(quadratic=qubo, constant=offset)
    operator, offset = cplex_program.to_ising()

    # ===========================================================
    # RUN QAOA
    optimizer = COBYLA(maxiter=max_iter)
    simulator = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=2048)
    qaoa = QAOA(operator, quantum_instance=simulator, optimizer=optimizer, p=p)
    result = qaoa.compute_minimum_eigenvalue()
    vector = sample_most_likely(result.eigenstate)
    best_sample = vector_to_sample(vector, map_names, variables)
    best_energy = bqm.energy(best_sample)

    row = {
        "best_sample": best_sample,
        "best_energy": qaoa.get_optimal_cost(),
        "best_energy_by_sample": best_energy,
        "p": p,
        "max_iter": max_iter
    }
    queue.put(row)


def run_qaoa_p1(_exp, _queue):
    run_qaoa_experiment(_exp, _queue, p=1, max_iter=1000)


def run_qaoa_p3(_exp, _queue):
    run_qaoa_experiment(_exp, _queue, p=3, max_iter=1000)


def run_qaoa_p5(_exp, _queue):
    run_qaoa_experiment(_exp, _queue, p=5, max_iter=1000)
