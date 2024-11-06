# https://github.com/jedwvv/QAOAvsVQE/blob/c6627bf46efcd596d7c15237635d352b3316f8f4/sprandom_init_QAOA.py
from time import time
import warnings
import numpy as np
import pickle as pkl
from parser_all import parse
from qiskit import Aer
from qiskit.utils.quantum_instance import QuantumInstance
from generate_qubos import solve_classically, arr_to_str
from classical_optimizers import NLOPT_Optimizer
from QAOA_methods import (CustomQAOA,
                         generate_points,
                         get_costs,
                         find_all_ground_states,
                         count_coupling_terms,
                         interp_point,
                         construct_initial_state,
                         n_qbit_mixer)
from QAOAEx import convert_from_fourier_point, convert_to_fourier_point
from qiskit_optimization import QuadraticProgram
            

# warnings.filterwarnings('ignore')

def main(args = None):
    """[summary]

    Args:
        raw_args ([type], optional): [description]. Defaults to None.
    """
    start = time()
    if args == None:
        args = parse()

    qubo_no = args["no_samples"]
    print_to_file("-"*50)
    print_to_file("QUBO_{}".format(qubo_no))
    #Load generated qubo_no
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], qubo_no), 'rb') as f:
        qubo, max_coeff, operator, offset, routes = pkl.load(f)
    qubo = QuadraticProgram()
    qubo.from_ising(operator)
    
    x_s, opt_value, classical_result = find_all_ground_states(qubo)
    print_to_file(classical_result)
    
    #Set optimizer method
    method = args["method"]
    optimizer = NLOPT_Optimizer(method = method, result_message=False)
    backend = Aer.get_backend("statevector_simulator")
    quantum_instance = QuantumInstance(backend = backend)

    approx_ratios = []
    prob_s_s = []
    p_max = args["p_max"]
    no_routes, no_cars = (args["no_routes"], args["no_cars"])

    custom = True
    if custom:
        initial_state = construct_initial_state(no_routes = no_routes, no_cars = no_cars)
        mixer = n_qbit_mixer(initial_state)
    else:
        initial_state, mixer = (None, None)

    fourier_parametrise = args["fourier"]
    print_to_file("-"*50)
    print_to_file("Now solving with QAOA... Fourier Parametrisation: {}".format(fourier_parametrise))
    for p in range(1, p_max+1):
        if p == 1:
            points = [[0,0]] + [ np.random.uniform(low = -np.pi/2+0.01, high = np.pi/2-0.01, size = 2*p) for _ in range(2**p)]
            next_point = []
        else:
            penalty = 0.6
            points = [next_point_l] + generate_points(next_point, no_perturb=min(2**p-1,10), penalty=penalty)
        construct_circ = False
        #empty lists to save following results to choose best result
        results = []
        exp_vals = []
        print_to_file("-"*50)
        print_to_file("    "+"p={}".format(p))
        optimizer.set_options(maxeval = 1000*p)
        for r, point in enumerate(points):
            qaoa_results, optimal_circ = CustomQAOA(operator,
                                                        quantum_instance,
                                                        optimizer,
                                                        reps = p,
                                                        initial_fourier_point= point,
                                                        initial_state = initial_state,
                                                        mixer = mixer,
                                                        construct_circ= construct_circ,
                                                        fourier_parametrise = fourier_parametrise,
                                                        qubo = qubo
                                                        )
            if r == 0:
                if fourier_parametrise:
                    next_point_l = np.zeros(shape = 2*p + 2)
                    next_point_l[0:p] = qaoa_results.optimal_point[0:p]
                    next_point_l[p+1:2*p+1] = qaoa_results.optimal_point[p:2*p]
                else:
                    next_point_l = interp_point(qaoa_results.optimal_point)
            exp_val = qaoa_results.eigenvalue * max_coeff
            exp_vals.append(exp_val)
            
            state_solutions = { item[0][::-1]: item[1:] for item in qaoa_results.eigenstate }
            
            for item in sorted(state_solutions.items(), key = lambda x: x[1][1], reverse = True)[0:5]:
                print_to_file( item )
                
            prob_s = 0
            for string in x_s:
                prob_s += state_solutions[string][1] if string in state_solutions else 0
            prob_s /= len(x_s) #normalise
            results.append((qaoa_results, optimal_circ, prob_s))
            print_to_file("    "+"Point_{}, Exp_val: {}, Prob_s: {}".format(r, exp_val, prob_s))
        minim_index = np.argmin(exp_vals)
        optimal_qaoa_result, optimal_circ, optimal_prob_s = results[minim_index]
        if fourier_parametrise:
            next_point = convert_from_fourier_point( optimal_qaoa_result.optimal_point, 2*p )
            next_point = convert_to_fourier_point( interp_point(next_point), 2*p + 2 )
#             next_point = np.zeros(shape = 2*p + 2)
#             next_point[0:p] = optimal_qaoa_result.optimal_point[0:p]
#             next_point[p+1:2*p+1] = optimal_qaoa_result.optimal_point[p:2*p]
        else:
            next_point = interp_point(optimal_qaoa_result.optimal_point)
        if construct_circ:
            print_to_file(optimal_circ.draw(fold=150))
        minim_exp_val = exp_vals[minim_index]
        approx_ratio = 1.0 - np.abs( (opt_value - minim_exp_val ) / opt_value )
        print_to_file("    "+"Minimum: {}, prob_s: {}, approx_ratio {}".format(minim_exp_val, optimal_prob_s, approx_ratio))
        approx_ratios.append(approx_ratio)
        prob_s_s.append(optimal_prob_s)
    print_to_file("-"*50)
    print_to_file("QAOA terminated")
    print_to_file("-"*50)
    print_to_file("Approximation ratios per layer: {}".format(approx_ratios))
    print_to_file("Prob_success per layer: {}".format(prob_s_s))
    save_results = np.append(approx_ratios, prob_s_s)
    if fourier_parametrise:
        with open('results_{}cars{}routes/RI_F_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print_to_file("Results saved in results_{}cars{}routes/RI_F_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    else:
        with open('results_{}cars{}routes/RI_NF_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print_to_file("Results saved in results_{}cars{}routes/RI_NF_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    finish = time()
    print_to_file("Time Taken: {}".format(finish - start))

def print_to_file(string, filepath = "RI_output.txt"):
    print(string)
    with open(filepath, 'a') as f:
        print(string, file=f)

if __name__ == "__main__":
    main()