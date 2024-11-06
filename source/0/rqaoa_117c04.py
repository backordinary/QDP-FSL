# https://github.com/jedwvv/QAOAvsVQE/blob/06a1f44b3f16d21c5c05801d590413a9a8ed5908/rqaoa.py
##Changes: Uses existing classical result if available, else solve
##Changes: Allows using baseline QAOA and custom QAOA by parsing -C to use custom initial state
##Changes: 

from time import time
import pickle as pkl
from copy import deepcopy
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.quantum_info import Statevector
import QAOAEx
from parser_all import parse
from generate_qubos import solve_classically, arr_to_str, str_to_arr, visualise_solution
from generate_problem import import_map
from classical_optimizers import NLOPT_Optimizer
from qiskit_optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from QAOA_methods import CustomQAOA, find_all_ground_states, QiskitQAOA
from pprint import pprint
import numpy as np
from qaoa import build_noise_model
import json

def main(args=None):
    start = time()
    if args == None:
        args = parse()
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'rb') as f:
        load_data = pkl.load(f)
        if len(load_data) == 5:
            qubo, max_coeff, operator, offset, routes = load_data
            classical_result = None
        else:
            qubo, max_coeff, operator, offset, routes, classical_result = load_data
    fourier_parametrise = args["fourier"]
    #Normalize qubo (Can recover original qubo via normalize_factor * qubo_objective )
    qubo, normalize_factor = reduce_qubo(qubo)
    if classical_result:
        classical_result._fval /= normalize_factor #Also normalize classical result
    

    #Noise
    if args["noisy"]:
        #Use symmetrised Hamiltonian and base QAOA
        args["symmetrise"] = True
        args["customise"] = False
        args["bias"] = False
        args["simulator"] = "aer_simulator_density_matrix"
        multiplier = args["multiplier"]
        print("Simulating with noise...Error mulitplier: {}".format(multiplier))
        with open('average_gate_errors.json', 'r') as f:
            noise_rates = json.load(f)
        noise_model = build_noise_model(noise_rates, multiplier)
    else:
        print("Simulating without noise...")
        noise_model = None
    
    #Initialize RQAOA object and make sure there is a classical solution
    rqaoa = RQAOA(qubo,
                  args["no_cars"],
                  args["no_routes"],
                  symmetrise = args["symmetrise"],
                  customise = args["customise"],
                  classical_result = classical_result,
                  simulator = args["simulator"],
                  noise_model = noise_model,
                  opt_str = args["optimizer"]
                 )
    print("Args: {}\n".format(args))
    iterate_time = time()
    print("First round of TQA-QAOA...")
    p = args["p_max"]
    qaoa_results = rqaoa.solve_qaoa(p, tqa=True, fourier_parametrise = fourier_parametrise) if p>1 else rqaoa.solve_qaoa(p, fourier_parametrise=fourier_parametrise)
    rqaoa.perform_substitution_from_qaoa_results(qaoa_results, biased = args["bias"])
    print("Performed variable substition(s).")
    num_vars = rqaoa.qubo.get_num_vars()
    print("Remaining variables: {}".format(num_vars))
    iterate_time_2 = time()
    print("Time taken (for this iteration): {}s".format(iterate_time_2 - iterate_time))
    
    t = 1
    #Recursively substitute variables to reduce qubit by one until 1 remaining
    while num_vars > 1:
        iterate_time = time()
        t+=1
        print( "\nRound {} of TQA-QAOA. Results below:".format(t) )
        qaoa_results = rqaoa.solve_qaoa(p, tqa=True, fourier_parametrise=fourier_parametrise) if p>1 else rqaoa.solve_qaoa(p, fourier_parametrise=fourier_parametrise)
        rqaoa.perform_substitution_from_qaoa_results(qaoa_results, biased = args["bias"])
        print("Performed variable substition(s).")
        num_vars = rqaoa.qubo.get_num_vars()
        print("Remaining variables: {}".format(num_vars))
        iterate_time_2 = time()
        print("Time taken (for this iteration): {}s".format(iterate_time_2 - iterate_time))

    print( "\nFinal round of QAOA. Eigenstate below:" )
    p=2
    points = [ [ np.pi * (np.random.rand() - 0.5) for _ in range(2*p) ] for _ in range(10) ]  + [ [ 0 for _ in range(2*p) ] ]
    qaoa_results = rqaoa.solve_qaoa( p, points = points )
    
    print( "\nProbabilities: {}".format(rqaoa.prob_s) )
    print( "Approx Qualities of (lowest_energy_state, most_probable_state): {}\n".format(rqaoa.approx_s) )

    if args["symmetrise"]:
        var_last = "X_anc" #Last variable should be ancilla if Hamiltonian was initially symmetrised
        rqaoa.var_values[var_last] = 0 #Now use the fact that ancilla should be in 0 state (so Z_anc = 1)
    else:
        max_state = sorted(qaoa_results.eigenstate, key = lambda x: x[2], reverse=True)[0]
        var_last = rqaoa.qubo.variables[0].name
        rqaoa.var_values[var_last] = int(max_state[0])
    
    #Now read solution from correlations and final value
    var_values = rqaoa.var_values
    replacements = rqaoa.replacements
    while True:
        for var, replacement in replacements.items():
            if replacement == None:
                continue
            elif replacement[0] in var_values and var not in var_values and replacement[1] != None:
                var_values[var] = var_values[ replacement[0] ] if replacement[1] == 1 else 1 - var_values[ replacement[0] ]
        if len(var_values.keys()) == args["no_cars"]*args["no_routes"]+1 and args["symmetrise"]: #1 extra value if using ancilla
            break
        elif len(var_values.keys()) == args["no_cars"]*args["no_routes"]:
            break
    
    #Remove Ancilla qubit in final result if it is in the variable values dict
    if "X_anc" in var_values:
        var_values.pop("X_anc") 
    
    list_values = list(var_values.values())
    cost = rqaoa.original_qubo.objective.evaluate(var_values)
    
    print(rqaoa.classical_result)
    print("\nRQAOA Solution: {}, Cost: {}".format(list_values, cost))
    
    #End of algorithm
    finish = time()
    print("\nTime taken: {} s".format(finish - start))
    
    #Naming of file to save results to
    if args["customise"] and not args["noisy"]: #Using custom QAOA
        if args["bias"]:
            filedir = 'results_{}cars{}routes_mps/Biased_RQAOA_{}_Cust_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"])
        elif args["symmetrise"]:
            filedir = 'results_{}cars{}routes_mps/Symmetrised_RQAOA_{}_Cust_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"])
        else:
            filedir = 'results_{}cars{}routes_mps/Regular_RQAOA_{}_Cust_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"])
    elif not args["customise"] and not args["noisy"]:
        if args["bias"]: #Using baseline QAOA
            filedir = 'results_{}cars{}routes_mps/Biased_RQAOA_{}_Base_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"])
        elif args["symmetrise"]:
            filedir = 'results_{}cars{}routes_mps/Symmetrised_RQAOA_{}_Base_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"])
        else:
            filedir = 'results_{}cars{}routes_mps/Regular_RQAOA_{}_Base_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"])
    elif args["noisy"]:
        filedir = 'results_{}cars{}routes/Noisy_S_RQAOA_{}_Base_p={}_Error={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"], args["multiplier"])
        
    
    #Save results to file
    save_results = np.append( rqaoa.prob_s, rqaoa.approx_s )
    with open(filedir, 'w') as f:
        np.savetxt(f, save_results, delimiter=',')
    
    result_saved_string = "Results saved in {}".format(filedir)
    print(result_saved_string)
    print("_"*len(result_saved_string))
    print("")
    
    ##END(main)
    
def reduce_qubo(qubo):
    #Perform reduction as many times possible up to some threshold
    max_coeff = np.max( np.append( qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array() ) )
    total_normalize_factor = 1.0 #To recover original qubo by multiplying this number to its objective
    qubo.objective.linear._coefficients = qubo.objective.linear._coefficients / max_coeff
    qubo.objective.quadratic._coefficients = qubo.objective.quadratic._coefficients / max_coeff
    qubo.objective.constant = qubo.objective.constant / max_coeff
    total_normalize_factor *= max_coeff
    return qubo, total_normalize_factor
    
class RQAOA:
    def __init__(self, qubo, no_cars, no_routes, **kwargs):
        opt_str = kwargs.get('opt_str', "LN_COBYLA")
        self.symmetrise = kwargs.get('symmetrise', False)
        self.customise = kwargs.get('customise', True)
        self.classical_result = kwargs.get("classical_result", None)
        simulator = kwargs.get("simulator", "aer_simulator_matrix_product_state")
        noise_model = kwargs.get("noise_model", None)
        
        if simulator == None:
            simulator = "aer_simulator_matrix_product_state"
        
        #Initializing other algorithm required objects
        var_list = qubo.variables
        if noise_model == None:
            self.quantum_instance = QuantumInstance( backend = Aer.get_backend(simulator), shots = 4096)
        elif noise_model:
            self.quantum_instance = QuantumInstance( backend = Aer.get_backend(simulator),
                                                     shots = 4096, 
                                                     noise_model = noise_model,
                                                     basis_gates = ["cx", "x", "sx", "rz", "id"]
                                                   )
                                                    
        self.random_instance = QuantumInstance(backend = Aer.get_backend("aer_simulator_matrix_product_state"), shots = 1000)
        #print backend name
        print("Quantum Instance: {}\n".format(self.quantum_instance.backend_name))
        #print backend name
        
        self.optimizer = NLOPT_Optimizer(opt_str)
        self.optimizer.set_options(max_eval = 1000)
        self.original_qubo = qubo
        self.qubo = qubo
        self.operator, self.offset = qubo.to_ising()
        
        #If no classical result, this will compute the appropriate self.classical_result, else this will simply re-organise already available result    
        self.solve_classically() 
         
        #Setup for variable replacements
        self.replacements = {var.name:None for var in var_list}
        self.no_cars = no_cars
        self.no_routes = no_routes
        self.car_blocks = np.empty(shape = (no_cars,), dtype=object)
        for car_no in range(no_cars):
            self.car_blocks[car_no] = ["X_{}_{}".format(car_no, route_no) for route_no in range(no_routes)]
        
        #Initialize variable placeholders in
        self.qaoa_result = None
        self.initial_state = None
        self.mixer = None
        self.benchmark_energy = None
        self.var_values = {}
        self.prob_s = []
        self.approx_s = []
        self.optimal_point = None
        
        #Random energy
        self.get_random_energy()
        
        #Symmetrise QUBO if required (after getting random energy WITHOUT ancilla)
        if self.symmetrise: 
            self.symmetrise_qubo()
            
        #Custom initial state and mixer if required
        if self.customise:
            if self.symmetrise:
                self.construct_initial_state(ancilla = True)
            else:
                self.construct_initial_state()
            self.construct_mixer()
        
    
    def construct_initial_state(self, ancilla=False):
        qc = QuantumCircuit()
        for car_no in range(self.no_cars):
            car_block = self.car_blocks[car_no]
            R = len(car_block)
            if R != 0 and R != 1:
                num_qubits = qc.num_qubits
                q_regs = QuantumRegister(R, 'car_{}'.format((car_no)))
                qc.add_register(q_regs)
                w_one = QuantumCircuit(q_regs)
                for r in range(0,R-1):
                    if r == 0:
                        w_one.ry(2*np.arccos(1/np.sqrt(R-r)),r)
                    elif r != R-1:
                        w_one.cry(2*np.arccos(1/np.sqrt(R-r)), r-1, r)
                for r in range(1,R):
                    w_one.cx(R-r-1,R-r)
                w_one.x(0)
                qc.append(w_one, range(num_qubits, num_qubits+R))
            elif R == 1:
                num_qubits = qc.num_qubits
                q_regs = QuantumRegister(R, 'car_{}'.format((car_no)))
                qc.add_register(q_regs)
                w_one = QuantumCircuit(q_regs)
                w_one.h(0)
                qc.append(w_one, range(num_qubits, num_qubits+R))
            else:
                continue
        qc = qc.decompose()
        self.initial_state = qc

        if ancilla:
            ancilla_reg = QuantumRegister(1, 'ancilla')
            self.initial_state.add_register(ancilla_reg)
            self.initial_state.h(ancilla_reg[0])    
    
    def symmetrise_qubo(self):
        new_operator = []
        operator, _ = self.qubo.to_ising()
        for op_1 in operator:
            coeff, op_1 = op_1.to_pauli_op().coeff, op_1.to_pauli_op().primitive
            op_1_str = op_1.to_label()
            Z_counts = op_1_str.count('Z')
            if Z_counts == 1:
                op_1_str = "Z" + op_1_str  #Add a Z in the last qubit to single Z terms
            else:
                op_1_str = "I" + op_1_str #Add an I in the last qubit to ZZ terms (no change in operator)
            pauli = PauliOp( primitive = Pauli(op_1_str), coeff = coeff )
            new_operator.append(pauli)
        symmetrised_qubo = QuadraticProgram()
        symmetrised_operator = sum(new_operator)
        symmetrised_qubo.from_ising(symmetrised_operator, self.offset, linear=True)
        self.qubo = symmetrised_qubo
        self.rename_qubo_variables()
        operator, _ = self.qubo.to_ising()
        self.operator = operator
        
    def rename_qubo_variables(self):
        original_qubo = self.original_qubo
        qubo = self.qubo
        variable_names = [ variable.name for variable in qubo.variables ]
        original_variable_names = [ (variable.name, 1) for variable in original_qubo.variables ]
        new_variable_names = original_variable_names + [("X_anc", 1)] if self.symmetrise else original_variable_names
        variables_dict = dict(zip(variable_names, new_variable_names))
        for new_variable_name in variables_dict.values():
            qubo.binary_var(name = new_variable_name[0])
        qubo = qubo.substitute_variables(variables = variables_dict)
        if qubo.status == QuadraticProgram.Status.INFEASIBLE:
            raise QiskitOptimizationError('Infeasible due to variable substitution')
        self.qubo = qubo

    def construct_mixer(self):
        from qiskit.circuit.parameter import Parameter
        initial_state = self.initial_state 
        no_qubits = initial_state.num_qubits
        t = Parameter('t')
        mixer = QuantumCircuit(no_qubits)
        mixer.append(initial_state.inverse(), range(no_qubits))
        mixer.rz(2*t, range(no_qubits))
        mixer.append(initial_state, range(no_qubits))
        self.mixer = mixer
    
    def solve_classically(self):
        if self.classical_result:
            print("There is an existing classical result. Using this as code proceeds.")
            print(self.classical_result)
            self.opt_value = self.classical_result.fval
        else:
            print("No classical result already available.")
            print("Now solving classically")
            _, opt_value, classical_result, _ = find_all_ground_states(self.original_qubo)
            self.classical_result = classical_result
            self.opt_value = opt_value
    
    def get_random_energy(self):
        #Get random benchmark energy for 0 layer QAOA (achieved by using layer 1 QAOA with [0,0] angles) and with only feasible states (custom initial state)
        self.construct_initial_state()
        self.construct_mixer()
        random_energy, _ = CustomQAOA(operator = self.operator,
                    quantum_instance = self.random_instance,
                    optimizer = self.optimizer,
                    reps = 1,
                    initial_state = self.initial_state,
                    mixer = self.mixer,
                    solve = False,
                    )
        #Remove custom initial state if using BASE QAOA
        self.initial_state = None
        self.mixer = None
        temp = random_energy
        self.random_energy = temp + self.offset
        if np.round( self.random_energy - self.opt_value, 6 ) < 1e-7:
            print("0 layer QAOA converged to exact solution. Shifting value up by |exact_ground_energy| instead to avoid dividing by 0 in approx quality.")
            self.random_energy += np.abs(self.random_energy)
        self.benchmark_energy = self.random_energy
        print("random energy: {}\n".format(self.random_energy))
    
    def solve_qaoa(self, p, **kwargs):
        if self.optimal_point is not None and 'point' not in kwargs:
            point = self.optimal_point
        else:
            point = kwargs.get("point", None)
        fourier_parametrise = kwargs.get("fourier_parametrise", False)
        self.optimizer.set_options(maxeval = 1000)
        tqa = kwargs.get('tqa', False)
        points = kwargs.get("points", None)
        symmetrised = self.symmetrise
        
        #Can sometimes end up with zero operator when substituting variables when we only have ZZ terms (symmetrised qubo),
        #e.g. if H = ZIZ (=Z1Z3 for 3 qubit system) and we know <Z1 Z3> = 1, so after substition H = II for the 2 qubit system.
        #H = II is then treated as an offset and not a Pauli operator, so the QUBO.to_ising() method returns a zero (pauli) operator.
        #In such cases it means the QUBO is fully solved and any solution will do, so chose "0" string as the solution. 
        #This also makes sure that ancilla bit is in 0 state. (we could equivalently choose any other string with ancilla in 0 state)
        def valid_operator(qubo):
            num_vars = qubo.get_num_vars()
            operator, _ = qubo.to_ising()
            valid = False
            operator = [operator] if isinstance(operator, PauliOp) else operator #Make a list if only one single PauliOp
            for op_1 in operator:
                coeff, op_1 = op_1.to_pauli_op().coeff, op_1.to_pauli_op().primitive
                if coeff >= 1e-6 and op_1 != "I"*num_vars: #if at least one non-zero then return valid ( valid = True )
                    valid = True
            return valid
        
        valid_op = valid_operator(self.qubo)
        num_vars = self.qubo.get_num_vars()

        if num_vars >= 1 and symmetrised and not valid_op:
            qaoa_results = self.qaoa_result
            qaoa_results.eigenstate = np.array( [ 1 ] + [ 0 ]*(2**num_vars - 1) )
            qaoa_results.optimizer_evals = 0
            qaoa_results.eigenvalue = self.qubo.objective.evaluate([0]*num_vars)
            qc = QuantumCircuit(num_vars)
            
        elif tqa:
            deltas = np.arange(0.45, 0.91, 0.05)
            point = np.append( [ (i+1)/p for i in range(p) ] , [ 1-(i+1)/p for i in range(p) ] )
            points = [delta*point for delta in deltas]
            if fourier_parametrise:
                points = [ QAOAEx.convert_to_fourier_point(point, len(point)) for point in points ]
            qaoa_results, _ = QiskitQAOA( self.operator,
                                                        self.quantum_instance,
                                                        self.optimizer,
                                                        reps = p,
                                                        initial_state = self.initial_state,
                                                        mixer = self.mixer,
                                                        fourier_parametrise = fourier_parametrise,
                                                        list_points = points,
                                                        qubo = self.qubo
                                                        )
        
        elif points is not None:
            if fourier_parametrise:
                points = [ QAOAEx.convert_to_fourier_point(point, len(point)) for point in points ]
            qaoa_results, _ = QiskitQAOA( self.operator,
                                                        self.quantum_instance,
                                                        self.optimizer,
                                                        reps = p,
                                                        initial_state = self.initial_state,
                                                        mixer = self.mixer,
                                                        fourier_parametrise = fourier_parametrise,
                                                        list_points = points,
                                                        qubo = self.qubo
                                                        )
 
        elif point is None:
            points = [ [0]*(2*p) ] + [ [ 2 * np.pi* ( np.random.rand() - 0.5 ) for _ in range(2*p)] for _ in range(10) ]
            qaoa_results, _ = QiskitQAOA( self.operator,
                                        self.quantum_instance,
                                        self.optimizer,
                                        reps = p,
                                        initial_state = self.initial_state,
                                        list_points = points,
                                        mixer = self.mixer,
                                        fourier_parametrise = fourier_parametrise,
                                        qubo = self.qubo
                                        )
        else:
            if fourier_parametrise:
                point =  QAOAEx.convert_to_fourier_point(point, len(point))
            qaoa_results, _ = QiskitQAOA( self.operator,
                                        self.quantum_instance,
                                        self.optimizer,
                                        reps = p,
                                        initial_state = self.initial_state,
                                        initial_point = point,
                                        mixer = self.mixer,
                                        fourier_parametrise = fourier_parametrise,
                                        qubo = self.qubo
                                        )
            
        point = qaoa_results.optimal_point
        eigenstate = qaoa_results.eigenstate
        if self.quantum_instance.is_statevector:
            from qiskit.quantum_info import Statevector
            eigenstate = Statevector(eigenstate)
            eigenstate = eigenstate.probabilities_dict()
        else:
            eigenstate = dict([(u, v**2) for u, v in eigenstate.items()]) #Change to probabilities
        num_qubits = len(list(eigenstate.items())[0][0])
        solutions = []
        eigenvalue = 0
        for bitstr, sampling_probability in eigenstate.items():
            bitstr = bitstr[::-1]
            value = self.qubo.objective.evaluate([int(bit) for bit in bitstr])
            eigenvalue += value * sampling_probability
            solutions += [(bitstr, value, sampling_probability)]
        qaoa_results.eigenstate = solutions
        qaoa_results.eigenvalue = eigenvalue

        self.optimal_point = point
        self.qaoa_result = qaoa_results


        #Sort states by increasing energy and decreasing probability
        sorted_eigenstate_by_energy = sorted(qaoa_results.eigenstate, key = lambda x: x[1])
        sorted_eigenstate_by_prob = sorted(qaoa_results.eigenstate, key = lambda x: x[2], reverse = True)
        
        #print energy-sorted state in a table
        self.print_state(sorted_eigenstate_by_energy)

        #Other print stuff
        print("Eigenvalue: {}".format(qaoa_results.eigenvalue))
        print("Optimal point: {}".format(qaoa_results.optimal_point))
        print("Optimizer Evals: {}".format(qaoa_results.optimizer_evals))
        scale = self.random_energy - self.opt_value
        approx_quality = np.round( (self.random_energy - sorted_eigenstate_by_energy[0][1])/ scale, 3 )
        approx_quality_2 = np.round( ( self.random_energy - sorted_eigenstate_by_prob[0][1] ) / scale, 3 )
        energy_prob = {}
        for x in qaoa_results.eigenstate:
            energy_prob[ np.round(x[1], 6) ] = energy_prob.get(np.round(x[1], 6), 0) + x[2]
        prob_s = np.round( energy_prob.get(np.round(self.opt_value, 6), 0), 6 )
        self.prob_s.append( prob_s )
        self.approx_s.append( [approx_quality, approx_quality_2] )
        print( "\nQAOA lowest energy solution: {}".format(sorted_eigenstate_by_energy[0]) )
        print( "Approx_quality: {}".format(approx_quality) )
        print( "\nQAOA most probable solution: {}".format(sorted_eigenstate_by_prob[0]) )
        print( "Approx_quality: {}".format(approx_quality_2) ) 

        return qaoa_results
    
    def print_state(self, eigenstate):
            header = '|'
            for var in self.qubo.variables:
                var_name = var.name
                header += '{:<5}|'.format(var_name.replace("_", ""))
            header += '{:<6}|'.format('Cost')
            header += '{:<5}|'.format("Prob")
            print("-"*len(header))
            print(header)
            print("-"*len(header))
            
            count = 0
            for item in eigenstate:
                string = '|'
                for binary_var in item[0]:
                    string += '{:<5} '.format(binary_var) #Binary string
                string += '{:<6} '.format(np.round(item[1], 2)) #Cost
                string += '{:<5}|'.format(np.round(item[2], 3)) #Prob
                print(string)
                
                #Print only first 20 states of lowest energy
                count += 1
                if count == 20: 
                    break
            
            print("-"*len(header))

    def perform_substitution_from_qaoa_results(self, qaoa_results, update_benchmark_energy=True, biased = True):
        
        correlations = self.get_biased_correlations(qaoa_results.eigenstate) if biased else self.get_correlations(qaoa_results.eigenstate)
        i, j = self.find_strongest_correlation(correlations)
        correlation = correlations[i, j]
        new_qubo = deepcopy(self.qubo)
        x_i, x_j = new_qubo.variables[i].name, new_qubo.variables[j].name
        if x_i == "X_anc":
            print("X_i was ancilla. Swapped")
            x_i, x_j = x_j, x_i #So ancilla qubit is never substituted out
            i, j = j, i #Also swap i and j
        print( "\nCorrelation: < {} {} > = {}".format(x_i.replace("_", ""), x_j.replace("_", ""), correlation)) 
        
        car_block = int(x_i[2])
#         #If same car_block and x_i = x_j, then both must be 0 since only one 1 in a car block
#         if x_i[2] == x_j[2] and correlation > 0 and len(self.car_blocks[car_block]) > 2: 
#             # set x_i = x_j = 0
#             new_qubo = new_qubo.substitute_variables({x_i: 0, x_j:0})
#             if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
#                 raise QiskitOptimizationError('Infeasible due to variable substitution {} = {} = 0'.format(x_i, x_j))
#             self.var_values[x_i] = 0
#             self.var_values[x_j] = 0
#             self.car_blocks[car_block].remove(x_i)
#             self.car_blocks[car_block].remove(x_j)
#             print("Two variable substitutions were performed due to extra information from constraints.")                    
        if correlation > 0: 
            # set x_i = x_j
            new_qubo = new_qubo.substitute_variables(variables={x_i: (x_j, 1)})
            if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                raise QiskitOptimizationError('Infeasible due to variable substitution {} = {}'.format(x_i, x_j))            
            self.replacements[x_i] = (x_j, 1)
            self.car_blocks[car_block].remove(x_i)
        else :
            # set x_i = 1 - x_j, this is done in two steps:
            # 1. set x_i = 1 + x_i
            # 2. set x_i = -x_j

            # 1a. get additional offset from the 1 on the RHS of (xi -> 1+xi)
            constant = new_qubo.objective.constant
            constant += new_qubo.objective.linear[i]
            constant += new_qubo.objective.quadratic[i, i]
            new_qubo.objective.constant = constant

            #1b get additional linear part from quadratic terms becoming linear due to the same 1 in the 1+xi as above
            for k in range(new_qubo.get_num_vars()):
                coeff = new_qubo.objective.linear[k]
                if k == i:
                    coeff += 2*new_qubo.objective.quadratic[i, k]
                else:
                    coeff += new_qubo.objective.quadratic[i, k]

                # set new coefficient if not too small
                if np.abs(coeff) > 1e-10:
                    new_qubo.objective.linear[k] = coeff
                else:
                    new_qubo.objective.linear[k] = 0

            #2 set xi = -xj
            new_qubo = new_qubo.substitute_variables(variables={x_i: (x_j, -1)})
            if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                raise QiskitOptimizationError('Infeasible due to variable substitution {} = -{}'.format(x_i, x_j))
            self.replacements[x_i] = (x_j, -1)
            self.car_blocks[car_block].remove(x_i)                
        
        self.qubo = new_qubo
        op, offset = new_qubo.to_ising() 
        self.operator = op
        self.offset = offset
        if self.customise:
            if self.symmetrise:
                self.construct_initial_state(ancilla = True)
            else:
                self.construct_initial_state()
            self.construct_mixer()
        
        
    def get_correlations(self, states) -> np.ndarray:
        """
        Get <Zi x Zj> correlation matrix from the eigenstate(state: Dict).

        Returns:
            A correlation matrix.
        """
        x, _, prob = states[0]
        n = len(x)
        correlations = np.zeros((n, n))
        for x, cost, prob in states:
            for i in range(n):
                for j in range(i):
                    if x[i] == x[j]:
                        correlations[i, j] += prob
                    else:
                        correlations[i, j] -= prob
        return correlations
    
    def get_z_zz(self, states) -> np.ndarray:
        """
        Get <Zi x Zj> correlation matrix from the eigenstate(state: Dict) along with < Zi > on the diagonal 

        Returns:
            A correlation matrix.
        """
        x, _, prob = states[0]
        n = len(x)
        z_zz_matrix = np.zeros((n, n))
        for x, cost, prob in states:
            for i in range(n):
                #Diagonal entries
                if x[i] == '0':
                    z_zz_matrix[i, i] += prob
                elif x[i] == '1':
                    z_zz_matrix[i, i] -= prob
                
                #Lower triangular entries
                for j in range(i):
                        if x[i] == x[j]:
                            z_zz_matrix[i, j] += prob
                        else:
                            z_zz_matrix[i, j] -= prob
                            
        return z_zz_matrix

    def find_strongest_z_or_zz(self, z_zz_matrix):

        # get absolute values and set diagonal to -1 to make sure maximum is always on off-diagonal
        abs_z_zz = np.abs(z_zz_matrix)

        # get index of maximum (by construction on off-diagonal)
        m_max = np.argmax(abs_z_zz.flatten())

        # translate back to indices
        i = int(m_max // len(z_zz_matrix))
        j = int(m_max - i*len(z_zz_matrix))

        return (i, j)

    def perform_substitution_from_qaoa_results_z(self, qaoa_results):
        
        z_zz_matrix = self.get_z_zz(qaoa_results.eigenstate)
        i, j = self.find_strongest_z_or_zz(z_zz_matrix)
        sign = z_zz_matrix[i, j]
        new_qubo = deepcopy(self.qubo)
        x_i, x_j = new_qubo.variables[i].name, new_qubo.variables[j].name
        if x_i == "X_anc":
            print("X_i was ancilla. Swapped")
            x_i, x_j = x_j, x_i #So ancilla qubit is never substituted out
            i, j = j, i #Also swap i and j
        print(z_zz_matrix)
        print( "\nMaximum: < {} {} > = {}".format(x_i.replace("_", ""), x_j.replace("_", ""), sign)) 
        print(new_qubo)
        
        car_block = int(x_i[2])            
        if i != j:
            if sign > 0: 
                # set x_i = x_j
                new_qubo = new_qubo.substitute_variables(variables={x_i: (x_j, 1)})
                if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                    raise QiskitOptimizationError('Infeasible due to variable substitution {} = {}'.format(x_i, x_j))            
                self.replacements[x_i] = (x_j, 1)
                self.car_blocks[car_block].remove(x_i)
            else:
                # set x_i = 1 - x_j, this is done in two steps:
                # 1. set x_i = 1 + x_i
                # 2. set x_i = -x_j

                # 1a. get additional offset from the 1 on the RHS of (xi -> 1+xi)
                constant = new_qubo.objective.constant
                constant += new_qubo.objective.linear[i]
                constant += new_qubo.objective.quadratic[i, i]
                new_qubo.objective.constant = constant

                #1b get additional linear part from quadratic terms becoming linear due to the same 1 in the 1+xi as above
                for k in range(new_qubo.get_num_vars()):
                    coeff = new_qubo.objective.linear[k]
                    if k == i:
                        coeff += 2*new_qubo.objective.quadratic[i, k]
                    else:
                        coeff += new_qubo.objective.quadratic[i, k]

                    # set new coefficient if not too small
                    if np.abs(coeff) > 1e-10:
                        new_qubo.objective.linear[k] = coeff
                    else:
                        new_qubo.objective.linear[k] = 0

                #2 set xi = -xj
                new_qubo = new_qubo.substitute_variables(variables={x_i: (x_j, -1)})
                if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                    raise QiskitOptimizationError('Infeasible due to variable substitution {} = -{}'.format(x_i, x_j))
                self.replacements[x_i] = (x_j, -1)
                self.car_blocks[car_block].remove(x_i)    
        else:
            if sign > 0: 
                # set x_i = 0
                try:
                    new_qubo = new_qubo.substitute_variables(constants={x_i: 0.0})
                except Exception as e:
                    print(e)
#                 if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
#                     raise QiskitOptimizationError('Infeasible due to variable substitution {} = {}'.format(x_i, 0))            
                self.replacements[x_i] = None
                self.var_values[x_i] = 0
                self.car_blocks[car_block].remove(x_i)
            else:
                #set x_i = 1
                try:
                    new_qubo = new_qubo.substitute_variables(constants={x_i: 1.0})
                except Exception as e:
                    print(e)
#                 if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
#                     raise QiskitOptimizationError('Infeasible due to variable substitution {} = {}'.format(x_i, 1))
                self.replacements[x_i] = None
                self.var_values[x_i] = 1
                self.car_blocks[car_block].remove(x_i)    
        
        self.qubo = new_qubo
        op, offset = new_qubo.to_ising() 
        self.operator = op
        self.offset = offset
        if self.customise:
            if self.symmetrise:
                self.construct_initial_state(ancilla = True)
            else:
                self.construct_initial_state()
            self.construct_mixer()    
              
    def get_biased_correlations(self, states) -> np.ndarray:
        """
        Get <Zi x Zj> correlation matrix from the eigenstate(states: Dict).

        Returns:
            A correlation matrix.
        """
        x, _, prob = states[0]
        n = len(x)
        correlations = np.zeros((n, n))
        for x, cost, prob in states:
            scaled_approx_quality = 1 / ( 1 + 2 ** (-self.benchmark_energy + cost) )
            for i in range(n):
                for j in range(i):
                    if x[i] == x[j]:
                        correlations[i, j] += scaled_approx_quality * prob
                    else:
                        correlations[i, j] -= scaled_approx_quality * prob
        return correlations

    def find_strongest_correlation(self, correlations):

        # get absolute values and set diagonal to -1 to make sure maximum is always on off-diagonal
        abs_correlations = np.abs(correlations)
        diagonal = np.diag( np.ones(len(correlations)) )
        abs_correlations = abs_correlations - diagonal

        # get index of maximum (by construction on off-diagonal)
        m_max = np.argmax(abs_correlations.flatten())

        # translate back to indices
        i = int(m_max // len(correlations))
        j = int(m_max - i*len(correlations))

        return (i, j)


if __name__ == '__main__':
    main()