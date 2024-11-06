# https://github.com/FedericoD94/BATQuO/blob/030a114bf082f00d37b9fdfca8c9e3d1c9cb5655/utils/qaoa_qiskit.py
import networkx as nx
from collections import Counter, defaultdict, namedtuple
from itertools import product
import random
import numpy as np
from scipy.stats import qmc

# QUANTUM
from qiskit import Aer, QuantumCircuit, execute
from qutip import *
from  utils.default_params import *

# VIZ
from matplotlib import pyplot as plt
from utils.default_params import *

class qaoa_qiskit(object):

    def __init__(self, G):
        self.G = G
        self.N = len(G)
        self.solution = self.classical_solution()
        self.gs_state = None
        self.gs_en = None
        self.deg = None
        


    def evaluate_cost(self,
                      string,
                      penalty=DEFAULT_PARAMS["penalty"]):
        '''
        configuration: eigenvalues
        '''
        configuration = np.array(tuple(string),dtype=int)
        cost = 0
        cost = -sum(configuration)
        for edge in self.G.edges:
            cost += penalty*(configuration[edge[0]]*configuration[edge[1]])
            
        return cost


    def classical_solution(self):
        '''
        Runs through all 2^n possible configurations and estimates how many max cliques there are and plots one
        '''
        results = {}

        eigen_configurations = list(product(['0','1'], repeat=len(self.G)))
        for eigen_configuration in eigen_configurations:
            single_string = "".join(eigen_configuration)
            results[single_string] = self.evaluate_cost(eigen_configuration)
            
        d = dict((k, v) for k, v in results.items() if v == np.min(list(results.values())))
        return d


    def quantum_algorithm(self,
                          params,
                          penalty=DEFAULT_PARAMS["penalty"]):
        '''
        Qiskit implementations of gates:
        Rz: https://qiskit.org/documentation/stubs/qiskit.circuit.library.RZGate.html
        Rzz: https://qiskit.org/documentation/stubs/qiskit.circuit.library.RZZGate.html
        Rx:
        from which we deduce the angles
        '''
        depth = int(len(params)/2)
        gammas = params[::2]
        betas = params[1::2]

        #INIZIALIZE CIRCUIT
        qc = QuantumCircuit(self.N, self.N)
        qc.h(range(self.N))

        #APPLY QAOA no. DEPTH TIMES
        for p in range(depth):
            for edge in self.G.edges():
                qc.rzz(gammas[p] * penalty / 2 , edge[0], edge[1])
                qc.rz(-gammas[p] * penalty / 2, edge[0])
                qc.rz(-gammas[p] * penalty / 2, edge[1])

            for i in self.G.nodes:
                qc.rz(gammas[p], i)

            qc.rx(2*betas[p], range(self.N))

        return qc

    def quantum_measure(self):

         #MEASURE
        meas = QuantumCircuit(self.N,self.N)
        meas.barrier(range(self.N))
        meas.measure(range(self.N), range(self.N))

        return meas

    def run_circuit(self, params, qc,
                    backend_name = 'qasm_simulator',
                    penalty=DEFAULT_PARAMS["penalty"],
                    shots=DEFAULT_PARAMS["shots"] ):

        backend = Aer.get_backend(backend_name)
        #The two seeds are necessary for reproducibility!  
        simulate = execute(qc, backend=backend, shots=shots, seed_transpiler = DEFAULT_PARAMS["seed"], seed_simulator = DEFAULT_PARAMS["seed"])
        results = simulate.result()

        return results

    def final_sampled_state(self,
                        params,
                        penalty=DEFAULT_PARAMS["penalty"],
                        shots=DEFAULT_PARAMS["shots"]):

        #MEASURE
        measure = self.quantum_measure()
        qc = self.quantum_algorithm(params)
        qc.compose(measure, inplace = True)

        results = self.run_circuit(params, qc, 'qasm_simulator', penalty, shots)
        counts = results.get_counts()

        #L'ordine va invertito perchè Qiskit usa little endian 
        pretty_counts = {k[::-1]:v for k,v in counts.items()}

        return pretty_counts

    def final_exact_state(self,
             params,
             penalty=DEFAULT_PARAMS["penalty"],
             shots=DEFAULT_PARAMS["shots"]):

        qc = self.quantum_algorithm(params)
        results = self.run_circuit(params, qc, "statevector_simulator",  penalty, shots = 1)
        st = results.get_statevector(qc)
        st.reshape((2**self.N,1))
        
        # Qiskit uses little endian convention (why????) so we need to invert the binary
        #combinations in order in the array, ex: 00111 becomes 11100 while 10101 stays 10101
        ordered_st = np.zeros(len(st), dtype = complex)
        for i, num in enumerate(st):
            string = '{0:b}'.format(i)
            invert_string = string[::-1]
            pos_in_array = int(invert_string, 2)
            ordered_st[pos_in_array] = num
        return ordered_st

    def circuit_unitary(self, params, penalty=DEFAULT_PARAMS["penalty"]):

        qc = self.quantum_algorithm(params)
        results = self.run_circuit(params, qc, 'unitary_simulator', penalty, shots = 1)
        uni = results.get_unitary(qc)

        return uni

    def expected_energy(self,
             params,
             shots=DEFAULT_PARAMS["shots"]):
        '''
        Applies QAOA circuit and estimates final energy
        '''

        counts = self.final_sampled_state(params)
        extimated_en = 0

        for configuration in counts:
            prob_of_configuration = counts[configuration]/shots
            extimated_en += prob_of_configuration * self.evaluate_cost(configuration)

        amplitudes =  np.fromiter(counts.values(), dtype=float)
        amplitudes = amplitudes / shots

        return extimated_en
    
    def expected_energy_and_variance(self,
             params,
             shots=DEFAULT_PARAMS["shots"]):
        '''
        Applies QAOA circuit and estimates final energy and variance
        '''

        counts = self.final_sampled_state(params)
        extimated_en = 0

        for configuration in counts:
            prob_of_configuration = counts[configuration]/shots
            extimated_en += prob_of_configuration * self.evaluate_cost(configuration)

        amplitudes = np.fromiter(counts.values(), dtype=float)
        amplitudes = amplitudes / shots

        return extimated_en, self.sample_variance(extimated_en, counts, shots)
    
    def sample_variance(self, sample_mean, counts, shots):
        estimated_variance = 0

        for configuration in counts:
            hamiltonian_i = self.evaluate_cost(configuration) # energy of i-th configuration
            estimated_variance += counts[configuration] * (sample_mean - hamiltonian_i)**2
        
        estimated_variance /= shots - 1 # use unbiased variance estimator

        return estimated_variance

    def plot_landscape(self,
                    param_range,
                    fixed_params = None,
                    num_grid=DEFAULT_PARAMS["num_grid"],
                    save = False):
        '''
        Plot energy landscape at p=1 (default) or at p>1 if you give the previous parameters in
        the fixed_params argument
        '''

        lin = np.linspace(param_range[0],param_range[1], num_grid)
        Q = np.zeros((num_grid, num_grid))
        Q_params = np.zeros((num_grid, num_grid, 2))
        for i, gamma in enumerate(lin):
            for j, beta in enumerate(lin):
                if fixed_params is None:
                    params = [gamma, beta]
                else:
                    params = fixed_params + [gamma, beta]
                Q[j, i] = self.expected_energy(params)
                Q_params[j,i] = np.array([gamma, beta])


        plt.imshow(Q, origin = 'lower', extent = [param_range[0],param_range[1],param_range[0],param_range[1]])
        plt.title('Grid Search: [{} x {}]'.format(num_grid, num_grid))
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)

        cb = plt.colorbar()
        plt.xlabel(r'$\gamma$', fontsize=20)
        plt.ylabel(r'$\beta$', fontsize=20)
        cb.ax.tick_params(labelsize=15)
        plt.show()

        if save:
            np.savetxt('../data/raw/graph_Grid_search_{}x{}.dat'.format(num_grid, num_grid), Q)
            np.savetxt('../data/raw/graph_Grid_search_{}x{}_params.dat'.format(num_grid, num_grid), Q)


    def plot_final_state_distribution(self, freq_dict):
        ''' Plots the final state given as a dictionary with {binary_strin:counts}'''

        sorted_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
        color_dict = {key: 'g' for key in sorted_freq_dict}
        for key in self.solution.keys():
            val = ''.join(str(key[i]) for i in range(len(key)))
            color_dict[val] = 'r'
        plt.figure(figsize=(12,6))
        plt.xlabel("configuration")
        plt.ylabel("counts")
        plt.xticks(rotation='vertical')
        plt.bar(sorted_freq_dict.keys(), sorted_freq_dict.values(), width=0.5, color = color_dict.values())

    def generate_random_points(self, N_points, depth, extreme_params, fixed_params=None, return_variance=False):
        X = []
        Y = []
        VAR = []
        np.random.seed(DEFAULT_PARAMS['seed'])
        random.seed(DEFAULT_PARAMS['seed'])
        
        hypercube_sampler = qmc.LatinHypercube(d=depth*2,  seed = DEFAULT_PARAMS['seed'])
        X =  hypercube_sampler.random(N_points)
        l_bounds = np.repeat(extreme_params[0], 2*depth)
        u_bounds = np.repeat(extreme_params[1], 2*depth)
        X = qmc.scale(X, l_bounds, u_bounds)
        X = X.tolist()
        for x in X:
            y, var_y = self.expected_energy_and_variance(x)
            Y.append(y)
            if return_variance:
                VAR.append(var_y)

        if return_variance:
            return X, Y, VAR
        else:
            return X, Y

    def list_operator(self, op):
        ''''
        returns a a list of tensor products with op on site 0, 1,2 ...
        '''
        op_list = []

        for qubit in range(self.N):
            op_list_i = []
            for m in range(self.N):
                op_list_i.append(qeye(2))

            op_list_i[qubit] = op
            op_list.append(tensor(op_list_i))

        return op_list

    def calculate_gs(self, penalty=DEFAULT_PARAMS["penalty"]):
        '''
        returns groundstate and energy
        '''

        sx_list = self.list_operator(sigmax())
        sz_list = self.list_operator(sigmaz())

        H=0
        for n in range(self.N):
            H +=  0.5 * sz_list[n]
        for i, edge in enumerate(self.G.edges):
            H += penalty/4 * sz_list[edge[0]]*sz_list[edge[1]]
            H -= penalty/4 * sz_list[edge[0]]
            H -= penalty/4 * sz_list[edge[1]]
        energies, eigenstates = H.eigenstates(sort = 'low') 
        _, degeneracies = np.unique(energies, return_counts = True)
        degeneracy = degeneracies[0]
        
        gs_en = energies[0]
        if degeneracy > 1:
            deg = degeneracy
            gs_state = eigenstates[:degeneracy]
        else:
            deg = degeneracy - 1
            gs_state = eigenstates[0]
            gs_state = gs_state.full()

        self.gs_state = gs_state
        self.gs_en = gs_en
        self.deg = deg

        return gs_en, gs_state, deg

    def fidelity_gs_exact(self, point):

        #calculate gs if it is not calculated
        if self.gs_state is None:
            self.gs_en, self.gs_state, self.deg = self.calculate_gs()

        fin_state_exact = self.final_exact_state(point)
        
        if self.deg:
            fidelities = [np.abs(np.dot(fin_state_exact, self.gs_state[i]))**2 for i in range(len(self.gs_state))]
            fidelity = np.sum(fidelities)
        else:
            fidelity = np.squeeze(np.abs(np.dot(fin_state_exact, self.gs_state))**2)

        return fidelity
        
    def fidelity_gs_sampled(self, x, solution_ratio = False):
        '''
        Fidelity sampled means how many times the solution(s) is measured
        '''
        #calculate gs if it is not calculated
        if self.gs_state is None:
            self.gs_en, self.gs_state, self.deg = self.calculate_gs()
            
        C = self.final_sampled_state(x)
        fid = 0
        for sol_key in self.solution.keys():
            fid += C[sol_key]
        
        fid = fid/DEFAULT_PARAMS['shots']
        
        if solution_ratio:
            sorted_dict = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
            first_key, second_key =  list(sorted_dict.keys())[:2]
            if (first_key in self.solution.keys()) and (second_key !=0):
                sol_ratio = C[first_key]/C[second_key]
            else:
                sol_ratio = 0
            return fid, sol_ratio
        else:
            return fid
            
    def solution_ratio(self, x):
        sol_ratio = 0
        
        C = self.final_sampled_state(x)
        
        sorted_dict = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
        first_key, second_key =  list(sorted_dict.keys())[:2]
        if (first_key in self.solution.keys()) and (second_key !=0):
            sol_ratio = C[first_key]/C[second_key]
        
        return sol_ratio
