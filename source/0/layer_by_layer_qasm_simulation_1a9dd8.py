# https://github.com/smitchaudhary/QAOA-MaxCut/blob/9a91a85cc16f49fa986ce1761e44b5edc132c676/layer_by_layer_qasm_simulation.py
# Private libraries
import qaoa_graphs as graphs
from layer_by_layer_qasm_functions import *

# General
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# Qiskit
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
import warnings
warnings.filterwarnings('ignore')

# Quantum Inspire
from quantuminspire.credentials import save_account
from quantuminspire.qiskit import QI

# Optimizers
from scipy.optimize import minimize, differential_evolution

# Noises
from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel, QuantumError, thermal_relaxation_error, depolarizing_error
from qiskit.test.mock import FakeVigo

# ---------------------Choose your fighter
G = graphs.erdos_renyi()

# ---------------------Choose the arena
# Simulators
backend = QasmSimulator()
#backend = QasmSimulator.from_backend(FakeVigo())

# Load IBM quantum experience
#provider = IBMQ.load_account()
#backend = provider.get_backend('ibmq_vigo')

# Load quantum inspire
#save_account('TOKEN') #saves account with my API token
#QI.set_authentication()
#backend = QI.get_backend('Starmon-5') #

shots = 10000

# In case we don't want a noise model:
noise_model = None

prev_gamma = [] ; prev_beta = []
cost_list = [] ; approx_ratio_list = [] ; calls_list = []

p_max = 5
for p in range(1,p_max + 1):

    # Initial values
    x0 = np.random.randn(2)

    # Parameters bounds (for SLQP and diff-evolution)
    bound = (0, 2*np.pi)
    bounds = []
    for i in range(2*p):
        bounds.append(bound)

    # Nelder-Mead optimizer:
    # max_expect_value = minimize(expect_value_function, x0=x0,args=(prev_gamma,prev_beta,backend,G,shots,p,noise_model), options={'disp': True, 'maxiter': 20000, 'maxfev' : 100000}, method = 'Nelder-Mead')

    # Differential evolution optimizer:
    max_expect_value = differential_evolution(expect_value_function,args=(prev_gamma,prev_beta,backend,G,shots,p,noise_model), bounds=bounds, maxiter = 10000, disp = False)

    optimal_gamma, optimal_beta = max_expect_value['x'][0], max_expect_value['x'][1]
    counts = execute_circuit(G, optimal_gamma, optimal_beta, prev_gamma, prev_beta, backend, shots, p, noise_model)
    number_of_calls, number_of_iterations = max_expect_value['nfev'], max_expect_value['nit']
    solution, solution_cost = get_solution(counts, G)
    avr_cost = -max_expect_value.get('fun')
    approx_ratio = avr_cost/solution_cost # Careful! This approximation ratio is computed assuming the algorithm is able to find
                                          # the solution (i.e. measures the state solution at least once). For enough shots (e.g. 10000) this is almost garanteed
    cost_list.append(avr_cost)
    approx_ratio_list.append(approx_ratio)
    calls_list.append(number_of_calls)
    prev_gamma.append(optimal_gamma)
    prev_beta.append(optimal_beta)

    print(f"Number of layers: {p}")
    print(f'Number of optimizer iterations: {number_of_iterations}')
    print(f'Number of calls to the objective function: {number_of_calls}')
    print('Optimal gamma, optimal beta = ', optimal_gamma, optimal_beta)
    print(f'The solution is the state {solution} with a cost value of {solution_cost}')
    print('Expectation value of the cost function = ', avr_cost)
    print('Approximation ratio = ', approx_ratio ) 
    print('  ')
    print(f'Number of calls list: {calls_list}')
    print(f"Average cost list: {cost_list}")
    print(f"Approximation ratio list: {approx_ratio_list}")
    print('-------------------------------------------------------------------')
    #plot_histogram(counts,figsize = (8,6),bar_labels = False)
    #plt.show()


#np.save('saved_cost_list', cost_list)
plt.plot(range(1,p_max+1), cost_list)
plt.show()
#plt.savefig('Cost list')


''' This is my attempt of building a cheating noise model 

# T1 and T2 values (from analysis)
T1 = 45e-3
T2 = 20e-3

# Gate execution times
T_1qubit_gates = 20e-6
T_2qubit_gates = 60e-6

# Thermal relaxation
single_qubit_error = thermal_relaxation_error(T1, T2, T_1qubit_gates) #single qubit gates
two_qubits_error = thermal_relaxation_error(T1, T2, T_2qubit_gates) #two qubit gates

#missing gate (in)fidelity error. 1qubit = 1.5 × 10 −3 , 2qubit = 4 × 10 −2

# Add errors to noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(two_qubits_error, ['Cx'])
'''


''' 
# Noise model: Depolarizing noise--> working
error1 = depolarizing_error(0.05, 1) #single qubit gates
error2 = depolarizing_error(0.05, 2) #two qubit gates

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error2, ['cx'])

#basis_gates=noise_model.basis_gates # We need to pass this as a parameter to the qiksit execute!
'''