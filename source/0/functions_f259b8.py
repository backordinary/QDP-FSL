# https://github.com/NTrevisani/DataScienceMaster/blob/b32ae615a4412143d6486ed767630f46ee4cfed4/TFM/MaxCut/Functions.py
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit import IBMQ
import pickle
import sys
import matplotlib.pyplot as plt 
import networkx as nx


# PI declaration
PI = np.pi


# The actual function
def qubo_to_ising(input_Q):

    # Define the 2x2 matrices we need

    # (1 + pauli_z)/2
    sigma_z = np.array([[1, 0], [0, 0]])

    # (1 - sigma_z)
    minus_z = np.array([[0, 0], [0, 1]])

    # Identity
    id_matrix = np.array([[1, 0], [0, 1]])

    n = len(input_Q)
    print("input:")
    print(input_Q)
    print("")
    
    # initialize H
    H = 0

    # compute the contribution of the i,j term to the Hamiltonian
    # i = left-side term = x_i (corresponds to sigma_z)
    for i in range(n):
        # j = right-side term = (1 - x_j) (corresponds to minus_z)
        for j in range(n):            
            # first term
            matrix_ij = 0
            if i == 0:
                matrix_ij = sigma_z
            elif j == 0:
                matrix_ij = minus_z
            else:
                matrix_ij = id_matrix
            
            # tensor product n times
            for k in range(1,n):
                if i == k:
                    new_term = sigma_z
                elif j == k:
                    new_term = minus_z
                else:
                    new_term = id_matrix                
                matrix_ij = np.kron(matrix_ij, new_term)

            # multiply by the i,j term of input_Q 
            matrix_ij = matrix_ij * input_Q[i,j]
            
            # sum
            H = H + matrix_ij
    
    return(-H)


# Compute the value of the cost function
# results: the results dictionary in the outcome of the circuit measurement
# weights: the original QUBO matrix
def cost_function_C(results, weights):
    
    # the eigenstates obtained by the evaluation of the circuit
    eigenstates = list(results.keys())
    
    # how many times each eigenstate has been sampled
    abundancies = list(results.values())
    
    # number of shots 
    shots = sum(results.values())
    
    # initialize the cost function
    cost = 0
    
    for k in range(len(eigenstates)):
        # ndarray of the digits extracted from the eigenstate string 
        x = np.array([int(num) for num in eigenstates[k]])
        # Cost function due to the k-th eigenstate
        cost = cost + x.dot(weights.dot(1-x)) * abundancies[k]
    
    return -cost / shots


# Write a VQE circuit as a parametric function
def VQE_circuit(theta, n, depth): 
    """Creates a variational-form RY ansatz.
    
    theta: (depth+1 x n) matrix of rotation angles,
    n: number of qbits,
    depth: number of layers.
    """
        
    if len(theta.ravel()) != ((depth+1) * n):        
        raise ValueError("Theta cannot be reshaped as a (depth+1 x n) matrix")

    theta.shape = (depth + 1, n)

    # Define the Quantum and Classical Registers
    q = QuantumRegister(n)
    c = ClassicalRegister(n)

    # Build the circuit for the ansatz
    circuit = QuantumCircuit(q, c)

    # Put all the qbits in the |+> state
    for i in range(n):
        circuit.ry(theta[0,i],q[i])
    circuit.barrier()
    
    # Now introduce the z-gates and RY-gates 'depth' times
    for j in range(depth):
        # Apply controlled-z gates
        for i in range(n-1):
            circuit.cz(q[i], q[i+1])

        # Introduce RY-gates
        for i in range(n):
            circuit.ry(theta[j+1,i],q[i])
        circuit.barrier()
    
    # Close the circuit with qbits measurements
    circuit.measure(q, c)
    
    return circuit    


# Write a QAOA circuit as a parametric function
def QAOA_circuit(gamma_beta, QUBO, depth): 
    """Creates a QAOA circuit.
    
    gamma_beta: input rotation angles.
    The first half of the variable contains the gamma angles
    and the second half the beta angles,
    QUBO: the original QUBO problem matrix,
    depth: number of layers.
    """
            
    # prepare the quantum and classical registers
    n_vertices = len(QUBO)

    if len(gamma_beta) != 2*depth + n_vertices:
        raise ValueError("'gamma_beta' parameter length must be equal to twice 'depth' parameter plus the number of vertices")

    gamma = gamma_beta[0:depth]
    beta  = gamma_beta[depth:2*depth]
    theta = gamma_beta[2*depth:2*depth + n_vertices]
    
    # Define the Quantum and Classical Registers
    q = QuantumRegister(n_vertices)
    c = ClassicalRegister(n_vertices)

    # Build the circuit
    circuit = QuantumCircuit(q, c)

    # apply the layer of Hadamard gates to all qubits
    #circuit.h(range(n_vertices))
    for i in range(n_vertices):
        circuit.ry(theta[i],q[i])
    circuit.barrier()
    
    # Repeat the circuit structure 'depth' times
    for k in range(depth):
        # apply the Ising type gates with angle gamma along the edges in E
        for i in range(n_vertices):
            for j in range(i,n_vertices):
                if QUBO[i,j] != 0:
                    circuit.cu1(-2*gamma[k]*QUBO[i,j], i, j)
                    circuit.u1(gamma[k], i)
                    circuit.u1(gamma[k], j)
    
        # then apply the single qubit X-rotations with angle beta to all qubits
        circuit.barrier()
        circuit.rx(2*beta[k], range(n_vertices))

    # Finally measure the result in the computational basis
    circuit.barrier()
    circuit.measure(range(n_vertices),range(n_vertices))
    
    return circuit    


# In case you want to use a real quantum device as backend
from qiskit import IBMQ

def cost_function_cobyla(params, 
                         weights,   # = W, 
                         n_qbits,   # = 5, 
                         depth,     # = 2,
                         shots,     # = 1024
                         cost,
                         algorithm    = "VQE", 
                         alpha        = 0.5,
                         backend_name = 'qasm_simulator',
                         verbosity    = False):
    """Creates a circuit, executes it and computes the cost function.
    
    params: ndarray with the values of the parameters to be optimized,
    weights: the original QUBO matrix of the problem,
    n_qbits: number of qbits of the circuit,
    depth: number of layers of the ciruit,
    shots: number of evaluations of the circuit state,
    cost: the cost function to be used. It can be: 
     - 'cost': mean value of all measured eigenvalues
     - 'cvar': conditional value at risk = mean of the
               alpha*shots lowest eigenvalues,
    alpha: 'cvar' alpha parameter
    verbosity: activate/desactivate some control printouts.
    
    The function calls 'VQE_circuit' to create the circuit, then
    evaluates it and compute the cost function.
    """
    
    if (verbosity == True):
        print("Arguments:")  
        print("params    = \n", params)
        print("weights   = \n", weights)
        print("qbits     = ", n_qbits)
        print("depth     = ", depth)
        print("shots     = ", shots)
        print("cost      = ", cost)
        print("algorithm = ", algorithm)
        print("alpha     = ", alpha)
        print("backend   = ", backend_name)
    
    if algorithm == "VQE":
        circuit = VQE_circuit(params, n_qbits, depth)
    elif algorithm == "QAOA":
        circuit = QAOA_circuit(params, weights, depth)
    
    if backend_name == 'qasm_simulator':
        backend = Aer.get_backend('qasm_simulator')
    else:
        provider = IBMQ.load_account()
        backend = provider.get_backend(backend_name)
    
    # Execute the circuit on a simulator
    job = execute(circuit, 
                  backend = backend, 
                  shots   = shots)
    results = job.result()
    
    if cost == 'cost':
        output = cost_function_C(results.get_counts(), weights)
    elif cost == 'cvar':
        output = cv_a_r(results.get_counts(), weights, alpha)
    else:
        raise ValueError("Please select a valid cost function")
    
    if (verbosity == True):
        print("cost = ", output)

    return output


# Is time a good figure of merit?
# https://stackoverflow.com/questions/27728483/understanding-the-output-of-scipy-optimize-basinhopping

import time
from scipy.optimize import minimize

def time_vs_shots(shots,
                  weights,
                  n_qbits,
                  depth,
                  backend_name,
                  final_eval_shots,
                  cost,
                  alpha = 0.5,
                  theta = 1,
                  algorithm = "VQE",
                  verbosity = False):
    """Returns the time taken to solve a VQE problem
    as a function of the shots.    
    
    Input parameters:
    shots: number of evaluations of the circuit state,
    weights: the original QUBO matrix of the problem,
    n_qbits: number of qbits of the circuit,
    depth: number of layers of the ciruit,
    backend_name: the name of the device where the optimization will be performed,
    final_eval_shots: number of shots for the evaluation of the optimized circuit,
    cost: the cost function to be used. It can be: 
     - 'cost': mean value of all measured eigenvalues
     - 'cvar': conditional value at risk = mean of the
               alpha*shots lowest eigenvalues,
    alpha: 'cvar' alpha parameter
    theta: the ansatz initial parameters. If set to 1, the 
        standard ry ansatz parameters are used,
    algorithm: the optimization algorithm to be used (VQE or QAOA),
    verbosity: activate/desactivate some control printouts.
    
    Output:
    elapsed_time: time taken for the optimization (in seconds)
    counts: dictionaty the results of the optimization
    shots: the 'shots' input parameter (it may be useful for analysis)
    n_func_evaluations: number of evaluations of the cost function
    final_eval_shots: shots for the optimal circuit evaluation
    optimal_angles: the theta parameters given by the optimization,
    final_cost: the cost function of the optimal circuit.
    
    """
    # Do this only if no initial parameters have been given
    if isinstance(theta, (int)):
        if algorithm == "VQE":
            # Create the rotation angles for the ansatz
            theta_0       = np.repeat(PI/2, n_qbits)
            theta_0.shape = (1, n_qbits)
            theta_1       = np.zeros((depth, n_qbits))
            theta         = np.concatenate((theta_0, theta_1), axis = 0) 
        elif algorithm == "QAOA":
            theta_0 = np.zeros(2*depth)
            theta_1 = np.repeat(PI/2, n_qbits)
            theta   = np.concatenate((theta_0, theta_1), axis = 0) 
    
    # Time starts with the optimization
    start_time = time.time()

    # Classical optimizer tuning
    res = minimize(fun     = cost_function_cobyla, 
                   x0      = theta.ravel(),     # the 'params' argument of 'cost_function_cobyla'
                   method  = 'COBYLA',          # we want to use the COBYLA optimization algorithm
                   options = {'maxiter': 500},  # maximum number of iterations
                   tol     = 0.0001,            # tolerance or final accuracy in the optimization 
                   args    = (weights, 
                              n_qbits, 
                              depth, 
                              shots,
                              cost,
                              algorithm,
                              alpha,
                              backend_name,
                              verbosity))    # the arguments of 'cost_function_cobyla', except 'params'

    # Time stops when the optimization stopshttps://qiskit.org/
    end_time = time.time()
    
    # Total time taken for the optimization
    elapsed_time = end_time - start_time 

    # Number of cost function evaluations during the optimization
    n_func_evaluations = res.nfev

    # Obtain the output distribution using the final parameters
    # VQE
    if algorithm == "VQE":
        optimal_circuit = VQE_circuit(res.x, 
                                      n_qbits, 
                                      depth)
    # QAOA
    elif algorithm == "QAOA":
        optimal_circuit = QAOA_circuit(res.x, 
                                       weights, 
                                       depth)

    # Define the backend for the evaluation of the optimal circuit
    # - in case it is a simulator
    if backend_name == 'qasm_simulator':
        backend = Aer.get_backend('qasm_simulator')
    # - in case it is a real quantum device
    else:
        provider = IBMQ.load_account()
        backend = provider.get_backend(backend_name)

    # Get the results from the circuit with the optimized parameters    
    counts = execute(optimal_circuit, 
                     backend, 
                     shots = final_eval_shots).result().get_counts(optimal_circuit)
    
    # The optimized rotation angles
    optimal_angles = res.x
    
    # The cost function of the optimal circuit
    final_cost = res.fun
    
    return elapsed_time, counts, shots, n_func_evaluations, final_eval_shots, optimal_angles, final_cost


# Plot function definition
from scipy.optimize import curve_fit
def scatter_plot(x, y, 
                 title = "", xlabel = "", ylabel = "", save_as = "", 
                 xlim = (-9999, -9999),
                 ylim = (-9999, -9999),
                 n_rep = 100,
                 do_fit = False, fit_func = 0,
                 x_err = 0, y_err = 0):
    # Plot declaration
    fig, ax = plt.subplots()
    #local_plot = ax.scatter(x = x,
    #                        y = y)

    # Now including error bars.
    local_plot = ax.errorbar(x, y,
                             yerr = y_err,
                             xerr = x_err,
                             fmt  = 'o')
    # Title
    ax.set_title(title)

    # Cosmetics
    if ylim == (-9999, -9999):
        ax.set_ylim(0.0, 1.5*np.max(y))
    else:    
        ax.set_ylim(ylim)
    if xlim == (-9999, -9999):
        ax.set_xlim(0.0, 1.1*np.max(x))
    else:    
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Fit
    if do_fit == True:
        if fit_func == 'sqrt':
            func = fit_sqrt
            fit_label = 'Square root fit'
        elif fit_func == 'pol0':
            func = fit_pol0
            fit_label = 'Constant fit'
        elif fit_func == 'pol1':
            func = fit_pol1
            fit_label = 'Linear fit'
        elif fit_func == 'pol2':
            func = fit_pol2
            fit_label = 'Quadratic fit'
        elif fit_func == 'pol3':
            func = fit_pol3
            fit_label = 'Grade 3 polynomial fit'
        elif fit_func == '1-exp':
            func = fit_1_exp
            fit_label = '1 - exp fit'
        else:
            raise ValueError("Please insert a valid fit function.")
        popt, pcov = curve_fit(func, x, y)
        print("Optimized parameters:") 
        print(popt)
        print("Parameters covariance:")
        print(pcov)
        plt.plot(x, func(x, *popt), 'g--',
                 label=fit_label)
        plt.legend()
          
    # Save as png and pdf
    if save_as != "":
        plt.savefig(save_as + '.png')
        plt.savefig(save_as + '.pdf')

    # Close
    plt.close()

        
# A few possible fitting functions
def fit_sqrt(x, a, b):
    return a + b * np.sqrt(x)

def fit_pol0(x, a):
    return a + 0*x

def fit_pol1(x, a, b):
    return a + b * x

def fit_pol2(x, a, b, c):
    return a + b*x + c*x*x

def fit_pol3(x, a, b, c, d):
    return a + b*x + c*x*x + d*x**3

def fit_1_exp(x, a, b):
    return a*(1 - np.exp(-b*x))  



# Compute the value of the cost function of each eigenstate in a solution
# and returns to 'best candidate' eigenstate
# results_dict: the eigenstate-freq dictionary returned by 'time_vs_shots'
# weights: the original QUBO matrix
def best_candidate_finder(results_dict, 
                          weights):
        
    # the eigenstates obtained by the evaluation of the circuit
    eigenstates = list(results_dict.keys())
        
    # initialize the cost function
    min_cost = 0
    best_candidate = 0
    
    for k in range(len(eigenstates)):
        # ndarray of the digits extracted from the eigenstate string 
        x = np.array([int(num) for num in eigenstates[k]])
        # Cost function of to the k-th eigenstate
        cost = x.dot(weights.dot(1-x))
        if cost > min_cost:
            min_cost = cost
            best_candidate = eigenstates[k]
    
    return best_candidate
        

# Function to compute F_opt
def F_opt_finder(results_obj,
                 n_shots,
                 weights,
                 opt_sol,
                 n_eigenstates = 1000):
    """Returns the fraction of optimal solutions.
    
    Given the object returned by 'time_vs_shots',
    computes the fraction of best_candidates solutions
    which are optimal solutions.
    
    Inputs:
    results_obj: the object returned by 'time_vs_shots',
    n_shots: the 'number of shots' to investigate,
    W: the original QUBO matrix,
    opt_sol: list of the optimal solutions to the problem,
    n_eigenstates: maximum number of eigenstates in a solution.
    """
    # Initialize the counter of repetitions for the
    # selected number of shots
    N_rep = 0
    # Initialize the counter of best candidates which 
    # are optimal solutions    
    N_bc  = 0
    # Scan all the entries of the object
    for res in results_obj:
        # Select only the entries corresponding to 
        # the selected number of shots
        if res[2] == n_shots:
            # If the number of shots is the one we want to check,
            # sum 1 to the number of repetitions
            N_rep += 1
            # Find best candidate
            bc = best_candidate_finder(res[1], weights)
            # best candidate must contain the optimal solution
            if bc in opt_sol:
                # best candidate must have less than 'n_eigenstates' eigenstates
                if len(res[1]) < n_eigenstates:
                    N_bc += 1
    # Initialize output value
    F_opt = 0
    # If N_rep is not 0, return the fraction of best candidates
    # which are also optimal solutions
    if N_rep != 0:
        F_opt = N_bc / N_rep
    else:
        print("The number of shots selected is not present")
    return F_opt


# CVaR definition
def cv_a_r(results, weights, alpha):
    """ The function computes the conditional value at risk of a solution.
    Inputs:
    results: the eigenstates-abundances dictionary returned by the optimization,
    weights: the original QUBO matrix,
    alpha: the parameter of CVaR. Alpha c (0,1] and represents the fraction of
    eigenstates considered in the computation.
    
    The computation of CVaR considers first the eigenstates associated to the lowest
    eigenvalues and moves to eigenstates associated to increasing eigenvalues.
    """
    # the eigenstates obtained by the evaluation of the circuit
    eigenstates = list(results.keys())

    # create ndarray of eigenvalues
    eigenvalues = np.array([])
    for k in range(len(eigenstates)):
        # ndarray of the digits extracted from the eigenstate string 
        x = np.array([int(num) for num in eigenstates[k]])
        eigenvalues = np.append(eigenvalues, -x.dot(weights.dot(1-x)))
    
    # number of shots 
    shots = sum(results.values())

    # Create a dataframe to manage all the variables used
    # Start from the 'results' dictionary
    cv_df = pd.DataFrame.from_dict(results, orient = 'index')
    cv_df.reset_index(level=0, inplace=True)
    cv_df.columns = ["eigenstate", "abundance"]
    cv_df["abundance"] = cv_df.abundance / shots
    
    # Adding eigenvalues
    cv_df['eigenvalue'] = eigenvalues
    
    # Sort by eigenvalues (smaller first)
    cv_df.sort_values(by = ["eigenvalue"], inplace = True)
    
    # Define 'cumulative eigenstate abundace': we want to sum
    # term until cumul_abund is less or equal to alpha
    cv_df["cumul_abund"] = np.cumsum(cv_df["abundance"])

    # We are overstimating the abundace of the last term of CVaR
    # by this quantity    
    diff = cv_df[cv_df["cumul_abund"] > 0.5].iloc[0].cumul_abund - alpha
    # corrected value
    new_abundance = cv_df[cv_df["cumul_abund"] > 0.5].iloc[0].abundance - diff
    
    # location of the value to be corrected
    qq = cv_df[cv_df["cumul_abund"] > 0.5].iloc[0]    
    # corrected abundance value
    cv_df.at[qq.name, "abundance"]   = new_abundance
    cv_df.at[qq.name, "cumul_abund"] = 0.5

    # Actual CVaR computation
    cv_df["cv"] = (cv_df["abundance"] * cv_df["eigenvalue"]) / alpha   
    cv_df["cumul_cv"] = np.cumsum(cv_df["cv"])

    # CVaR value
    cvar = cv_df.at[qq.name, "cumul_cv"]
    
    return cvar


# We can save the results to produce them once and analyze them later
#https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
def save_object(obj, filename):
    with open(filename, 'wb') as output:  
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)   
        print("Object saved as", filename)


# This function is useful to check the memory usage
# https://stackoverflow.com/a/1094933/1870254
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# Comparison of a list of quantities on the same canvas
def plot_comparison(x, y, legend, leg_loc = "upper right",
                    title = "", xlabel = "", ylabel = "", save_as = "", 
                    xlim = (-9999, -9999),
                    ylim = (-9999, -9999),
                    x_err = [], y_err = [],
                    leg_col = 2):
    """Compares a list of plots on the same canvas.
        
    The x and the y must be manually defined as a list, as
    the corresponding legend labels.
    """
    fig, ax = plt.subplots()

    print("Printing {0}".format(save_as))
    
    if x_err == []:
        x_err = np.zeros(len(x))
    if y_err == []:
        y_err = np.zeros(len(y))
    
    # First plot
    for nplot in range(len(y)):
        #local_plot1 = ax.scatter(x     = x[nplot],
        #                         y     = y[nplot],
        #                         label = legend[nplot])
        # Now including error bars.
        local_plot1 = ax.errorbar(x     = x[nplot],
                                  y     = y[nplot],
                                  label = legend[nplot],
                                  yerr  = y_err[nplot],
                                  xerr  = x_err[nplot],
                                  fmt   = 'o')

    # Title
    ax.set_title(title)

    # Cosmetics
    if ylim == (-9999, -9999):
        ax.set_ylim(0.0, 1.5*np.max(y))
    else:    
        ax.set_ylim(ylim)
    ax.set_xlim(0.0, 1.1*np.max(x))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc = leg_loc, ncol = leg_col)

    # Save as png and pdf
    if save_as != "":
        plt.savefig(save_as + '.png')
        plt.savefig(save_as + '.pdf')

    # Close
    plt.close()

        
# Define a random graph and return the corresponding QUBO matrix
def random_graph_producer(n_vert, n_edge, seed = 2000, verbosity = False):
    """Produces a random graph with n_vert vertices and n_edge edges.
    
    Returns the matrix associated to the graph generated.
    If verbosity is True, the graph is drawn and the associated
    matrix is printed.
    """
    # Set the seed to the input value
    np.random.seed(seed)

    # Random graph production 
    G = nx.gnm_random_graph(n    = n_vert, 
                            m    = n_edge,
                            seed = seed)

    # QUBO matrix definition
    Q = np.zeros([n_vert, n_vert])

    for i in range(n_vert):
        for j in range(i):
            temp = G.get_edge_data(i, j, default = 0)
            if temp != 0:
                Q[i,j] = np.random.randint(1,11)
    Q += Q.T

    if verbosity == True:
        print(Q)
        nx.draw_networkx(G)
        
    return Q


# Solve a Max-Cut problem using brute force approach
def brute_force_solver(Q, verbosity = False):
    """Solve a Max-Cut problem using brute force approach.
    
    Returns the solutions as a list of strings. 
    If verbosity is set to true, the graph is plotted
    with the two subsets of vertices painted with different 
    colors.
    """
    # The graph associated to the QUBO matrix
    G = nx.from_numpy_matrix(Q)
    # Initialize best cost function value
    best_cost_brute = 0 
    # Get matrix shape
    n = Q.shape[0]
    # computing all possible combinations
    # initialize output
    xbest_brute = []
    # store all the eigenvalues to get mean and std-dev
    eigenvalues = np.array([])
    for b in range(2**n):
        # x stores all the 2^n possible combinations of 0 and 1
        # for a vector of length n 
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
        # initialize cost function value
        cost = 0
        # scan all possible costs and keep the highest one
        # (now we want to maximize our score!)
        for i in range(n):
            for j in range(n):
                cost = cost + Q[i,j]*x[i]*(1 - x[j])
        # store the cost function value as a eigenvalue
        eigenvalues = np.append(eigenvalues, cost)
        if cost > best_cost_brute:
            xbest_brute = [x] 
            best_cost_brute = cost
        elif cost == best_cost_brute:
            xbest_brute.append(x) 
    
    # Showing results    
    if verbosity == True:    
        colors = ['r' if xbest_brute[0][i] == 0 else 'b' for i in range(n)]
        nx.draw_networkx(G, node_color = colors)
        print('\nBest solution = ' + str(xbest_brute) + ' cost = ' + str(best_cost_brute)) 
    
    # Transform the solution in a list of strings
    for res in range(len(xbest_brute)):
        xbest_brute[res] = ''.join(map(str, xbest_brute[res]))

    return xbest_brute, best_cost_brute, eigenvalues


# Load results from pickle file and prepare them for analysis
def load_files(file_name, shot_list):
    """Load results from pickle file and prepare them for analysis.
    
    file_name: the initial part of the file name:
    e.g. if the file to load is called "Scan_10qbits_128.pkl",
    then file_name = 'Scan_10qbits'
    shots_list: a list of the number of shots you want to load.
    
    The function returns a list of structured objects containing:
    - optimization time; 
    - dictionary of results {'eigenstate', normalized_frequency};
    - number of shots used for the optimization;
    - number of evaluation of the cost function;
    - number of shots used for the evaluation of the optimal circuit,
    - the optimized parameters (ansatza rotation angles),
    - the optimal circuit cost function.
    """
    
    # List of results. Every entry corresponds to a given file.
    # Each file contains all the simulations of a given number of shots
    scan = []
    
    # Actually load all the selected files for a chunk of optimizations
    for shot in shot_list:
        # Prepare the complete file name 
        load_file_name = "{0}_{1}shots.pkl".format(file_name, str(shot))
        # Actually load the results and append them to the list
        with open(load_file_name, 'rb') as input:
            for pick in pickle.load(input): 
                scan.append(pick)
    
    # Normalize results for plotting
    for res in scan:
        for key, value in res[1].items():
            res[1][key] = res[1][key] / res[4]

    return scan


# Analyzes the results loaded by 'load_files' function
def analyze_results(scan, 
                    shot_list, 
                    weights, 
                    brute_solution, 
                    cost_function,
                    alpha = 1):
    """Analyzes the results loaded by 'load_files' function.
    
    scan: the object returned by 'load_files';
    shots_list: a list of the number of shots you want to analyze;
    weights: the original QUBO matrix;
    brute_solution: the optimal solution computed using brute 
    force approach;
    cost_function: the cost function used in the optimization
    ('cost' or 'cvar');
    alpha: in case the cost function is 'cvar', the CVaR
    alpha parameter used in the optimization.
    
    The function returns a 
    """
    # fraction of solution containing the optimal solution
    frac_list = np.array([])
    for shot in shot_list:
        frac = F_opt_finder(scan, shot, weights, brute_solution)
        frac_list = np.append(frac_list, frac)
        
    # Prepare the results so that it is easier to plot them
    
    # Create list of optimization times
    ntimes = np.array([])
    for i in range(len(scan)):
        ntimes = np.append(ntimes, scan[i][0])

    # Create list of nfev (number of cost function evaluations)
    nfevs = np.array([])
    for i in range(len(scan)):
        nfevs = np.append(nfevs, scan[i][3])

    # Create list of shots
    nshots = np.array([])
    for i in range(len(scan)):
        nshots = np.append(nshots, scan[i][2])

    # Create list of number of eigenstates in the solution
    neigenst = np.array([])
    for i in range(len(scan)):
        neigenst = np.append(neigenst, len(scan[i][1]))

    # Create list of optimized parameters
    ntheta = []
    for i in range(len(scan)):
        ntheta.append(scan[i][5])
        
    # Create list of cost function values
    ncost = np.array([])
    for i in range(len(scan)):
        ncost = np.append(ncost, scan[i][6])

    # Put the lists in a dataframe
    df = pd.DataFrame(list(zip(ntimes, nfevs, nshots, neigenst, ncost, ntheta)), 
               columns = ['time', 'nfevs', 'shots', 'eigenstates', 'cost', 'theta'])

    # Add the total number of circuit evaluation
    df['ncircevs'] = df['nfevs'] * df['shots']

    # Group by shots and average
    df_plot = df.groupby(['shots']).mean()
    df_plot.reset_index(level=0, inplace=True)
    df_plot["frac"] = frac_list
    
    # Group by shots and take standard deviation - used for cost function only
    df_std_dev = df.groupby(['shots']).std()
    df_std_dev.reset_index(level=0, inplace=True)
    # Add cost function std-dev to df_plot
    df_plot["cost_std_dev"] = df_std_dev["cost"]    
    
    return df, df_plot


# QUBO matrix used as example
W = np.array([[0, 1, 2, 0, 0],
              [1, 0, 2, 0, 0],
              [2, 2, 0, 2, 2],
              [0, 0, 2, 0, 1],
              [0, 0, 2, 1, 0]])


# groupby and select min
# df_min_mean = df_mean.groupby('shots')['cost']
# df_mean = df_mean.assign(min_cost=df_min_mean.transform(min))
# df_mean.groupby('shots')
# df_mean[df_mean['cost'] == df_mean['min_cost']]
