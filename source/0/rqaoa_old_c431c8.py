# https://github.com/jedwvv/QAOAvsVQE/blob/ebb0e02f8f589dacd711c1817be629bcf464715f/.rqaoa_old.py
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
from QAOA_methods import CustomQAOA, find_all_ground_states
import numpy as np

def main(args=None):
    start = time()
    if args == None:
        args = parse()
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'rb') as f:
        load_data = pkl.load(f)
        if len(load_data) == 5:
            qubo, max_coeff, operator, offset, routes = load_data
        else:
            qubo, max_coeff, operator, offset, routes, classical_result = load_data
    
    #Normalize qubo (Can recover original qubo via normalize_factor * qubo_objective )
    qubo, normalize_factor = reduce_qubo(qubo)
    
    #Initialize RQAOA object
    rqaoa = RQAOA(qubo, args["no_cars"], args["no_routes"], symmetrise = args["symmetrise"])
    
    print("Args: {}".format(args))
    print("First round of TQA-QAOA...")
    p = args["p_max"]
    qaoa_results = rqaoa.solve_qaoa(p, tqa=True)
    rqaoa.perform_substitution_from_qaoa_results(qaoa_results, biased = args["bias"])
    print("Performed variable substition(s) and constructed new initial state and mixer.")
    num_vars = rqaoa.qubo.get_num_vars()
    print("Remaining variables: {}".format(num_vars))
    t = 1
    
    #Recursively substitute variables to reduce qubit by one until 1 remaining
    while num_vars > 1:
        t+=1
        qaoa_results = rqaoa.solve_qaoa(p, tqa=True)
        print( "Round {} of TQA-QAOA. Results below:".format(t) )
        rqaoa.perform_substitution_from_qaoa_results(qaoa_results, biased = args["bias"])
        print("Performed variable substition(s) and constructed new initial state and mixer.")
        num_vars = rqaoa.qubo.get_num_vars()
        print("Remaining variables: {}".format(num_vars))

    print( "Final round of QAOA Done. Eigenstate below:" )
    p=2
    points = [ [ np.pi * (np.random.rand() - 0.5) for _ in range(2*p) ] for _ in range(10) ]  + [ [ 0 for _ in range(2*p) ] ]
    qaoa_results = rqaoa.solve_qaoa( p, points = points )
    
    print( "Probabilities: {}".format(rqaoa.prob_s) )
    print( "Approx Qualities of (lowest_energy_state, most_probable_state): {}".format(rqaoa.approx_s) )

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
    
    print(rqaoa.result)
    print(var_values)
    print("{}, Cost: {}".format(list_values, cost))
    
    #Save results
    save_results = np.append( rqaoa.prob_s, rqaoa.approx_s )
    if args["bias"]:
        with open('results_{}cars{}routes_mps/Biased_RQAOA_{}_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
            print("Results saved in results_{}cars{}routes_mps/Biased_RQAOA_{}_p={}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"]))
    
    elif args["symmetrise"]:
        with open('results_{}cars{}routes_mps/Symmetrised_RQAOA_{}_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
            print("Results saved in results_{}cars{}routes_mps/Symmetrised_RQAOA_{}_p={}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"]))
    
    else:
        with open('results_{}cars{}routes_mps/Regular_RQAOA_{}_p={}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
            print("Results saved in results_{}cars{}routes_mps/Regular_RQAOA_{}_p={}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"]))
    
    finish = time()
    print("Time taken: {} s".format(finish - start))

def reduce_qubo(qubo):
    #Perform reduction as many times possible up to some threshold
    max_coeff = np.max( np.append( qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array() ) )
    total_normalize_factor = 1.0 #To recover original qubo by multiplying this number to its objective
    while np.round(max_coeff) > 15.0:
        factor = np.round(np.log(max_coeff))
        qubo.objective.linear._coefficients = qubo.objective.linear._coefficients / factor
        qubo.objective.quadratic._coefficients = qubo.objective.quadratic._coefficients / factor
        qubo.objective.constant = qubo.objective.constant / factor
        total_normalize_factor *= factor
        max_coeff = np.max( np.append( qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array() ) )
    return qubo, total_normalize_factor 
    
class RQAOA:
    def __init__(self, qubo, no_cars, no_routes, **kwargs):
        opt_str = kwargs.get('opt_str', "LN_SBPLX")
        self.symmetrise = kwargs.get('symmetrise', False)
        var_list = qubo.variables
        self.optimizer = NLOPT_Optimizer(opt_str)
        self.optimizer.set_options(max_eval = 1000)
        self.original_qubo = qubo
        self.qubo = qubo
        self.solve_classically()
        op, offset = qubo.to_ising()
        self.operator = op
        self.offset = offset
        if self.symmetrise:
            self.symmetrise_qubo()
        self.quantum_instance = QuantumInstance(backend = Aer.get_backend("aer_simulator_matrix_product_state"), shots = 4096)
        self.replacements = {var.name:None for var in var_list}
        self.no_cars = no_cars
        self.no_routes = no_routes
        self.car_blocks = np.empty(shape = (no_cars,), dtype=object)
        for car_no in range(no_cars):
            self.car_blocks[car_no] = ["X_{}_{}".format(car_no, route_no) for route_no in range(no_routes)]
        self.qaoa_result = None
        self.benchmark_energy = None
        self.var_values = {}
        self.construct_initial_state()
        self.construct_mixer()
        self.get_random_energy()
        self.get_benchmark_energy()
        self.prob_s = []
        self.approx_s = []
        self.optimal_point = None
    
    def construct_initial_state(self):
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

        if self.symmetrise:
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
        for (variable_name, new_variable_name) in variables_dict.items():
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
        x_s, opt_value, classical_result, _ = find_all_ground_states(self.original_qubo)
        self.result = classical_result
        x_arr = classical_result.x
        self.x_s =  [ x_str[::-1] for x_str in x_s ]
        self.opt_value = opt_value
    solve
    def get_random_energy(self):
        #Get random benchmark energy for 0 layer QAOA (achieved by using layer 1 QAOA with [0,0] angles)
        random_energy, _ = CustomQAOA(operator = self.operator,
                    quantum_instance = self.quantum_instance,
                    optimizer = self.optimizer,
                    reps = 1,
                    initial_state = self.initial_state,
                    mixer = self.mixer,
                    solve = False,
                    )
        temp = random_energy
        self.random_energy = temp + self.offset
        if np.round( self.random_energy - self.opt_value, 6 ) < 1e-7:
            print("0 layer QAOA converged to exact solution. Shifting value up by |exact_ground_energy| instead to avoid dividing by 0 in approx quality.")
            self.random_energy += np.abs(self.random_energy)
        print("random energy: {}".format(self.random_energy))
            
    def get_benchmark_energy(self):
        #Get benchmark energy with 0-layer QAOA (just as random_energy)
        benchmark_energy, _ = CustomQAOA(operator = self.operator,
                    quantum_instance = self.quantum_instance,
                    optimizer = self.optimizer,
                    reps = 1,
                    initial_state = self.initial_state,
                    mixer = self.mixer,
                    solve = False,
                    )
        temp = benchmark_energy + self.offset
        #Choose minimum of benchmark_energy if there already exists self.benchmark_energy
        self.benchmark_energy = min(self.benchmark_energy, temp) if self.benchmark_energy else temp
        return self.benchmark_energy
    
    def solve_qaoa(self, p, **kwargs):
        if self.optimal_point and 'point' not in kwargs:
            point = self.optimal_point
        else:
            point = kwargs.get("point", None)
        fourier_parametrise = True
        self.optimizer.set_options(maxeval = 1000)
        tqa = kwargs.get('tqa', False)
        points = kwargs.get("points", None)
        symmetrised = self.symmetrise
        
        #Can sometimes end up with zero operator when substituting variables when we only have ZZ terms (symmetrised qubo),
        #e.g. if H = ZIZ (=Z1Z3 for 3 qubit system) and we know <Z1 Z3> = 1, so after substition H = II for the 2 qubit system.
        #H = II is then treated as an offset and not a Pauli operator, so the QUBO results to a zero (pauli) operator.
        #In such cases it means the QUBO is fully solved and any solution will do, so chose "0" string as the solution. 
        #This also makes sure that ancilla bit is in 0 state. (we could equivalently choose something like "100" instead the "000" for 3 remaining variables)solve
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
            qaoa_results.eigenstate = [ ('0'*num_vars, self.qubo.objective.evaluate([0]*num_vars), 1) ]
            qaoa_results.optimizer_evals = 0
            qaoa_results.eigenvalue = self.qubo.objective.evaluate([0]*num_vars)
            qc = QuantumCircuit(num_vars)
            
        elif tqa:
            deltas = np.arange(0.45, 0.91, 0.05)
            point = np.append( [ (i+1)/p for i in range(p) ] , [ 1-(i+1)/p for i in range(p) ] )
            points = [delta*point for delta in deltas]
            fourier_parametrise = True
            if fourier_parametrise:
                points = [ QAOAEx.convert_to_fourier_point(point, len(point)) for point in points ]
            qaoa_results, _ = CustomQAOA( self.operator,
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
            fourier_parametrise = True
            if fourier_parametrise:
                points = [ QAOAEx.convert_to_fourier_point(point, len(point)) for point in points ]
            qaoa_results, _ = CustomQAOA( self.operator,
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
            list_points = [0]*(2*p) + [ [ 2 * np.pi* ( np.random.rand() - 0.5 ) for _ in range(2*p)] for _ in range(5) ]
            fourier_parametrise = True
            if fourier_parametrise:
                points = [ QAOAEx.convert_to_fourier_point(point, len(point)) for point in points ]
            qaoa_results, _ = CustomQAOA( self.operator,
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
            fourier_parametrise = True
            if fourier_parametrise:
                point =  QAOAEx.convert_to_fourier_point(point, len(point))
            qaoa_results, _ = CustomQAOA( self.operator,
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
        qaoa_results.eigenvalue = sum( [ x[1] * x[2] for x in qaoa_results.eigenstate ] )
        self.optimal_point = QAOAEx.convert_to_fourier_point(point, len(point)) if fourier_parametrise else point
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
        scale = self.random_energy - self.result.fval
        approx_quality = np.round( (self.random_energy - sorted_eigenstate_by_energy[0][1])/ scale, 3 )
        approx_quality_2 = np.round( ( self.random_energy - sorted_eigenstate_by_prob[0][1] ) / scale, 3 )
        energy_prob = {}
        for x in qaoa_results.eigenstate:
            energy_prob[ np.round(x[1], 6) ] = energy_prob.get(np.round(x[1], 6), 0) + x[2]
        prob_s = np.round( energy_prob.get(np.round(self.result.fval, 6), 0), 6 )
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
            for item in eigenstate:
                string = '|'
                for binary_var in item[0]:
                    string += '{:<5} '.format(binary_var) #Binary string
                string += '{:<6} '.format(np.round(item[1], 2)) #Cost
                string += '{:<5}|'.format(np.round(item[2], 3)) #Prob
                print(string)
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
        
#         #If only one remaining variable and all other variables are 0, then remaining must be 1.
#         check = sum( [self.var_values.get("X_{}_{}".format(car_block, route_no), 0) for route_no in range(self.no_routes)] )
#         if len(self.car_blocks[car_block]) == 1 and check == 0: 
#             x_r = self.car_blocks[car_block][0] #remaining variable
#             new_qubo = new_qubo.substitute_variables({x_r: 1})
#             if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
#                 raise QiskitOptimizationError('Infeasible due to variable substitution {} = 1'.format(x_r))
#             self.car_blocks[car_block].remove(x_r)
#             print("{} = 1 can also be determined from all other variables being 0 for car_{}".format(x_r, car_block))
        
        #Update variable eliminated QUBO
        self.qubo = new_qubo
        op, offset = new_qubo.to_ising() 
        self.operator = op
        self.offset = offset
        self.construct_initial_state()
        self.construct_mixer()
#         if update_benchmark_energy:
#             temp = self.get_benchmark_energy()
        
        
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