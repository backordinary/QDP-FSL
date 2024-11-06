# https://github.com/VicentePerezSoloviev/EDA_QAOA/blob/2ea08641bbf249d4b34b36f845a90927dbed0553/optimization.py
import numpy as np
import pandas as pd

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, ADAM, CG, AQGD, GSLS, GradientDescent, SPSA
import pickle
import random
import time

random.seed(1234)

with open('max_cut_12.pkl', 'rb') as file:
    max_cut = pickle.load(file)

qubit_op, _ = max_cut.get_operator()

optimizers = [COBYLA(maxiter=100), L_BFGS_B(maxiter=100), SLSQP(maxiter=100),
              ADAM(maxiter=100), CG(maxiter=100), AQGD(maxiter=100),
              GradientDescent(maxiter=100), SPSA(maxiter=100)]
optimizers_names = ['COBYLA', 'L_BFGS_B', 'SLSQP', 'ADAM', 'CG', 'AQGD', 'GradientDescent', 'SPSA']

converge_cnts = np.empty([len(optimizers)], dtype=object)
converge_vals = np.empty([len(optimizers)], dtype=object)

iterations = range(10)
ps = range(1, 12)
index = 0
filename = 'output_optimizers_12.csv'
dt = pd.DataFrame(columns=['opt', 'it', 'p', 'best_cost', 'time'])

for i, optimizer in enumerate(optimizers):

    for p in ps:
        for it in iterations:
            start_time = time.process_time()

            counts = []
            values = []

            def store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)


            vqe = QAOA(optimizer, callback=store_intermediate_result,
                       quantum_instance=QuantumInstance(backend=Aer.get_backend('statevector_simulator')), reps=p)
            result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
            finish_time = time.process_time()

            converge_cnts[i] = np.asarray(counts)
            converge_vals[i] = np.asarray(values)

            dt.loc[index] = [optimizers_names[i], it, p, min(converge_vals[i]), finish_time-start_time]
            dt.to_csv(filename)
            index = index + 1

            # TODO: save the optimum parameters
