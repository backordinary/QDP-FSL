# https://github.com/koenmesman/qaoa/blob/773cc96d0e310ea7b0cf408cd542bcc2bc13fee2/python/application_test.py
#########################################################################
# QPack                                                                 #
# Koen Mesman, TU Delft, 2021                                           #
# This file runs the QAOA instance and writes the output data           #
#########################################################################
from math import pi, acos, sqrt
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from qaoa_def import cost_tsp
from math import pi, acos
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator
import Problem_set as ps

plt.interactive(True)
#backend = Aer.get_backend('qasm_simulator')
from qiskit.circuit.library import RYGate, RZZGate, OR, ZGate
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
import nlopt
import numpy as np
import qiskit.converters.circuit_to_gate
from numpy import *


plt.interactive(True)
from scipy.optimize import minimize
import nlopt
import numpy as np
import Classic_opt as co
from qaoa_def import qaoa_landscape, tsp_circ, dicke_init, dsp_cost
from data_gen import generate_graph, test_graph, tsp_graph
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
backend = Aer.get_backend('qasm_simulator')

qaoa_shots = 10000


def write_data(save_data, data_name):
    data_path = 'Test_data/' + data_name + '_' + str(qaoa_shots) + '_timing_data.dat'

    #t_data = np.loadtxt('Test_data/empty_file.dat')
    t_data = [[], []]
    try:
        load_data = np.loadtxt(data_path)
        save_data = np.array(save_data)
        t_data = np.hstack((load_data, save_data))


    except IOError:
        t_data = np.array(save_data)
        #print('data after IOERROR')
        #np.savetxt(data_path, 0)

    np.savetxt(data_path, t_data)
    print("data saved for " + data_name)


def run_qaoa(q_app, opt_func, graph, p):
    init_param = [pi, pi]
    result = opt_func(init_param, graph, p, q_app)
    return result

def data_test(q_app, problem_range, data_shots, opt_func, p):
    d_time = []
    data_x = []
    for problem_size in problem_range:  # loop over problem size
        ratio = 1
        # generate a (pseudo) random graph
        graph = {
            'max-cut': generate_graph(problem_size, ratio),
            'tsp': tsp_graph(problem_size, ratio),
            'dsp': generate_graph(problem_size, ratio)
        }.get(q_app)

        for d in range(data_shots):  # loop over data points per problem size, time optimizer
            print('start run ', d, 'for ', q_app)
            start = time.time()
            [result, res_param] = run_qaoa(q_app, opt_func, graph, p)
            d_time.append(time.time() - start)
            print(result)
            print("now finished size ", problem_size, " number ", d)
            data_x.append(problem_size)

            new_data = [data_x, d_time]
        write_data(new_data, q_app)

#data_test('dsp', range(10, 15), 50, shgo_fun, 1)
#data_test('max-cut', range(20, 23), 30, shgo_fun, 1)
#data_test('tsp', range(5, 6), 10, shgo_fun, 1)
#data_test('dsp', range(22, 25), 1, shgo_fun, 1)
#data_test('max-cut', range(15, 25), 40, shgo_fun, 1)
n=6
graph = [n, ps.regular_graph(n)]
result = run_qaoa('max-cut', co.nm, graph, 1)
print(result)
