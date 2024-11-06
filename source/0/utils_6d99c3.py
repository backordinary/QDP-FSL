# https://github.com/anaghasethuraman01/Delivery-Routing-System-using-Quantum-Computing/blob/d597e57680cd8b13b6eb2bba5cd034ecdbd36748/DeliveryRoutingSystem/backend/utils.py
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
#cplex
import cplex
from cplex.exceptions import CplexError
# Qiskit packages
from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import NumPyMinimumEigensolver, QAOA, VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from vrp_problem import VRPProblem
from vrp_solvers import DBScanSolver
from vrp_solvers import FullQuboSolver
from server import db
import DWaveSolvers

# def getConnectionDetails(): 
    
def getRoute(n,k,algo):
    if algo == 'vqe':
        return vqe(n,k)
    elif algo == 'qaoa':
        return qaoa(n,k)
    elif algo == 'cplex':
       return classical(n,k)
    elif algo == 'DBScan':
       return DWaveSolver(n,k, "dbscan")
    elif algo == 'FullQubo':
       return DWaveSolver(n,k, "fullqubo")
def getRandomNodesFromDb(n):
    nodeMap = {}
    locations = list(db.locations.find().limit(n))
    print("Locations: ", locations)
    addresses = []
    for loc in locations:
        addresses.append(loc['location'])
    # addresses = ['Los Angeles', 'Sacramento', 'Charlotte', 'San Jose', 'San Diego']
    #addresses = ['San Jose','Milpitas', 'Palo Alto', 'Fremont']
    # addresses = ['Milpitas', 'Palo Alto', 'San Jose', 'Fremont']
    geolocator = Nominatim(user_agent="VRP Using QC")
    xc = np.zeros([n])
    yc = np.zeros([n])
    for i in range(0, n):
        location = geolocator.geocode(addresses[i])
        nodeMap[i] = addresses[i]
        xc[i] = location.latitude
        yc[i] = location.longitude
    
    instance = np.zeros([n, n])
    for ii in range(0, n):
        for jj in range(ii + 1, n):
            instance[ii, jj] = math.sqrt((xc[ii] - xc[jj]) ** 2 + (yc[ii] - yc[jj]) ** 2)
            instance[jj, ii] = instance[ii, jj]

    # print('Input nodes:\n',instance)

    return xc, yc, instance, nodeMap
    
def cplex_solution(n,K,instance):
    # refactoring
    my_obj = list(instance.reshape(1, n**2)[0])+[0. for x in range(0,n-1)]
    my_ub = [1 for x in range(0,n**2+n-1)]
    my_lb = [0 for x in range(0,n**2)] + [0.1 for x in range(0,n-1)]
    my_ctype = "".join(['I' for x in range(0,n**2)]) + "".join(['C' for x in range(0,n-1)])

    my_rhs = 2*([K] + [1 for x in range(0,n-1)]) + [1-0.1 for x in range(0,(n-1)**2-(n-1))] + [0 for x in range(0,n)]
    my_sense = "".join(['E' for x in range(0,2*n)]) + "".join(['L' for x in range(0,(n-1)**2-(n-1))])+"".join(['E' for x in range(0,n)])

    try:
        my_prob = cplex.Cplex()
        populatebyrow(n,my_prob,my_obj,my_ub,my_lb,my_ctype,my_sense,my_rhs)

        my_prob.solve()

    except CplexError as exc:
        print('Encountered error in cplex:',exc)
        return

    x = my_prob.solution.get_values()
    x = np.array(x)
    cost = my_prob.solution.get_objective_value()

    return x,cost


def populatebyrow(n,prob,my_obj,my_ub,my_lb,my_ctype,my_sense,my_rhs):
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.variables.add(obj = my_obj, lb = my_lb, ub = my_ub, types = my_ctype)

    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)

    rows = []
    for ii in range(0,n):
        col = [x for x in range(0+n*ii,n+n*ii)]
        coef = [1 for x in range(0,n)]
        rows.append([col, coef])

    for ii in range(0,n):
        col = [x for x in range(0+ii,n**2,n)]
        coef = [1 for x in range(0,n)]

        rows.append([col, coef])

    # Sub-tour elimination constraints:
    for ii in range(0, n):
        for jj in range(0,n):
            if (ii != jj)and(ii*jj>0):

                col = [ii+(jj*n), n**2+ii-1, n**2+jj-1]
                coef = [1, 1, -1]

                rows.append([col, coef])

    for ii in range(0,n):
        col = [(ii)*(n+1)]
        coef = [1]
        rows.append([col, coef])

    prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rhs)

def binary_representation(n,K,instance, x_sol=0):
    A = np.max(instance) * 100  # A parameter of cost function

    # Determine the weights w
    instance_vec = instance.reshape(n ** 2)
    w_list = [instance_vec[x] for x in range(n ** 2) if instance_vec[x] > 0]
    w = np.zeros(n * (n - 1))
    for ii in range(len(w_list)):
        w[ii] = w_list[ii]

    # Some variables I will use
    Id_n = np.eye(n)
    Im_n_1 = np.ones([n - 1, n - 1])
    Iv_n_1 = np.ones(n)
    Iv_n_1[0] = 0
    Iv_n = np.ones(n - 1)
    neg_Iv_n_1 = np.ones(n) - Iv_n_1

    v = np.zeros([n, n * (n - 1)])
    for ii in range(n):
        count = ii - 1
        for jj in range(n * (n - 1)):

            if jj // (n - 1) == ii:
                count = ii

            if jj // (n - 1) != ii and jj % (n - 1) == count:
                v[ii][jj] = 1.0

    vn = np.sum(v[1:], axis=0)

    # Q defines the interactions between variables
    Q = A * (np.kron(Id_n, Im_n_1) + np.dot(v.T, v))

    # g defines the contribution from the individual variables
    g = (
        w
        - 2 * A * (np.kron(Iv_n_1, Iv_n) + vn.T)
        - 2 * A * K * (np.kron(neg_Iv_n_1, Iv_n) + v[0].T)
    )

    # c is the constant offset
    c = 2 * A * (n - 1) + 2 * A * (K ** 2)

    try:
        max(x_sol)
        # Evaluates the cost distance from a binary representation of a path
        fun = (
            lambda x: np.dot(np.around(x), np.dot(Q, np.around(x)))
            + np.dot(g, np.around(x))
            + c
        )
        cost = fun(x_sol)
    except:
        cost = 0

    return Q, g, c, cost



def construct_problem(Q, g, c, qubit_needed,n):
    qp = QuadraticProgram()
    for i in range(n * (n - 1)):
        qp.binary_var(str(i))
    qp.objective.quadratic = Q
    qp.objective.linear = g
    qp.objective.constant = c
    qp.objective.num_qubits=qubit_needed
    return qp

def get_optimizer(algo,quantum_instance):
    optimizer = None
    if algo == 'vqe':
        vqe = VQE(quantum_instance=quantum_instance)
        optimizer = MinimumEigenOptimizer(min_eigen_solver=vqe)
    elif algo == 'qaoa':
        qaoa = QAOA(quantum_instance=quantum_instance)
        optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
    return optimizer

def solve_problem(qp,algo,n,k,instance,):
    algorithm_globals.random_seed = 10598
    quantum_instance = QuantumInstance(
        BasicAer.get_backend("qasm_simulator"),
        seed_simulator=algorithm_globals.random_seed,
        seed_transpiler=algorithm_globals.random_seed,
    )


    optimizer = get_optimizer(algo,quantum_instance)
    result = optimizer.solve(qp)
    # compute cost of the obtained result
    _, _, _, level = binary_representation(n,k,instance,x_sol=result.x)
    return result, level

def checkBinaryRepresentation(z):
    # Check if the binary representation is correct
    try:
        if z is not None:
            Q, g, c, binary_cost = binary_representation(x_sol=z)
            print("Binary cost:", binary_cost, "classical cost:", classical_cost)
            if np.abs(binary_cost - classical_cost) < 0.01:
                print("Binary formulation is correct")
            else:
                print("Error in the binary formulation")
        else:
            print("Could not verify the correctness, due to CPLEX solution being unavailable.")
            Q, g, c, binary_cost = quantum_optimizer.binary_representation()
            print("Binary cost:", binary_cost)
    except NameError as e:
        print("Warning: Please run the cells above first.")
        print(e)

def visualize_solution(xc, yc, x, C, n, K, title_str, nodeMap):
    plt.figure()
    plt.scatter(xc, yc, s=200)
    for i in range(len(xc)):
        plt.annotate(nodeMap[i], (xc[i] + 0.15, yc[i]), size=16, color='r')
    plt.plot(xc[0], yc[0], 'r*', ms=20)

    plt.grid()

    for ii in range(0, n ** 2):

        if x[ii] > 0:
            ix = ii // n
            iy = ii % n
            plt.arrow(xc[ix], yc[ix], xc[iy] - xc[ix], yc[iy] - yc[ix], length_includes_head=True, head_width=0.010)

    plt.title(title_str+' cost = ' + str(int(C * 100) / 100.))
    plt.show()

def vqe(n,k):
    print('**********************vqe implementation**********************')
    qubit_needed = n*(n-1)
    xc, yc, instance, nodeMap = getRandomNodesFromDb(n)
    #get classical result
    x,z,classical_cost = get_classical_solution(n,k,instance)
    Q, g, c, binary_cost = binary_representation(n,k,instance, x_sol=z)
    qp = construct_problem(Q, g, c,qubit_needed,n)
    quantum_solution, quantum_cost = solve_problem(qp,'vqe',n,k,instance)
    print(quantum_solution, quantum_cost)
    x_quantum = np.zeros(n**2)
    kk = 0
    for ii in range(n ** 2):
        if ii // n != ii % n:
            x_quantum[ii] = quantum_solution[kk]
            kk +=  1


    # visualize the solution
    # visualize_solution(xc, yc, x_quantum, quantum_cost, n, k, 'Quantum', nodeMap)
    x_quantum_2d = get_traversed_path(x_quantum,n)
    solution = []
    solution.append(x_quantum_2d)
    new_xc, new_yc = get_new_coord(xc,yc, solution, n)
    return new_xc, new_yc, x_quantum_2d, quantum_cost, nodeMap, qubit_needed

def get_new_coord(xc, yc, traversed_path, n):
    new_xc = []
    new_yc = []
    print('This is the route',traversed_path)
    for i in range(0, len(traversed_path)):
        print("Vehicle: ", i+1)
        new_xc.append([])
        new_yc.append([])
        for j in range(0, len(traversed_path[i])):
            node_number = traversed_path[i][j]
            new_xc[i].append(xc[node_number])
            new_yc[i].append(yc[node_number])

    #for i in range(n):
    #    node_number = traversed_path[i]
    #    new_xc.append(xc[node_number])
    #    new_yc.append(yc[node_number])
    print(new_xc)
    print(new_yc)
    return new_xc, new_yc


def get_traversed_path(x_quantum, n):
    print(len(x_quantum))
    print(x_quantum)
    x_quantum_2d = x_quantum.reshape(n,n)
    
    print(x_quantum_2d)
    next = None
    curr = 0
    traversed_path = []
    while next != 0:
        traversed_path.append(curr)
        for i in range(n):
            if x_quantum_2d[curr][i] == 1:
                next = i
                break
        curr = next
    return traversed_path

def qaoa(n,k):
    print('**********************qaoa implementation**********************')
    qubit_needed = n*(n-1)
    xc, yc, instance, nodeMap = getRandomNodesFromDb(n)
    #get classical result
    x,z,classical_cost = get_classical_solution(n,k,instance)
    Q, g, c, binary_cost = binary_representation(n,k,instance, x_sol=z)
    qp = construct_problem(Q, g, c,qubit_needed,n)
    quantum_solution, quantum_cost = solve_problem(qp,'qaoa',n,k,instance)
    print(quantum_solution, quantum_cost)
    x_quantum = np.zeros(n**2)
    kk = 0
    for ii in range(n ** 2):
        if ii // n != ii % n:
            x_quantum[ii] = quantum_solution[kk]
            kk +=  1

    print(x_quantum)
    # visualize the solution

    # visualize_solution(xc, yc, x_quantum, quantum_cost, n, k, 'Quantum', nodeMap)
    x_quantum_2d = get_traversed_path(x_quantum,n)
    solution = []
    solution.append(x_quantum_2d)
    new_xc, new_yc = get_new_coord(xc,yc, solution, n)
    return new_xc, new_yc, x_quantum_2d, quantum_cost, nodeMap, qubit_needed

def DWaveSolver(n,k, algo):
    print(f'**********************Dwave {algo} implementation**********************')
    nodeMap = {}
    sources = [0]
    destinations = np.zeros((n-1), dtype=int)
    for i in range(1, n):
        destinations[i-1] = i
    locations = list(db.locations.find().limit(n))
    print("Locations: ", locations)
    addresses = []
    for loc in locations:
        addresses.append(loc['location'])
    #addresses = ['San Jose', 'Menlo Park', 'Sunnyvale', 'Cupertino', 'Milpitas', 'Palo Alto']
    nodes_num = len(sources) + len(destinations)

    # Parameters for solve function.
    only_one_const = 10000000.
    order_const = 1.

    # Reading weights of destinations.
    weights = np.zeros((nodes_num), dtype=int)

    # Reading costs.

    geolocator = Nominatim(user_agent="VRP Using QC")
    xc = np.zeros([nodes_num])
    yc = np.zeros([nodes_num])
    for i in range(0, nodes_num):
        location = geolocator.geocode(addresses[i])
        nodeMap[i] = addresses[i]
        xc[i] = location.latitude
        yc[i] = location.longitude
    
    costs = np.zeros((nodes_num, nodes_num))
    for ii in range(0, nodes_num):
        for jj in range(ii + 1, nodes_num):
            costs[ii, jj] = math.sqrt((xc[ii] - xc[jj]) ** 2 + (yc[ii] - yc[jj]) ** 2)
            costs[jj, ii] = costs[ii, jj]

    print('Input nodes:\n',costs)

    # Reading vehicles.
    vehicles = k
    capacities = np.ones((vehicles), dtype=int)

    problem = VRPProblem(sources, costs, capacities, destinations, weights) 
    # Solving problem on SolutionPartitioningSolver.
    if algo == "dbscan":
        solver = DBScanSolver(problem, anti_noiser = False, max_len = 25)
    if algo == "fullqubo":
        solver = FullQuboSolver(problem)
    solution = solver.solve(only_one_const, order_const, solver_type = 'qpu')

    # Checking if solution is correct.
    if solution == None or solution.check() == False:
        print("Solver hasn't find solution.\n")

    result = []
    print("Solution : ", solution.solution) 
    for i in range(0, len(solution.solution)):
        print("Vehicle: ", i+1)
        result.append([])
        for j in range(0, len(solution.solution[i])-1):
            print(addresses[solution.solution[i][j]], end ="--")
            result[i].append(solution.solution[i][j])
        print("\n")  
    print("Result: ", result)   
    print("Total cost : ", solution.total_cost())
    print("\n")

    # visualize_solution(xc, yc, x_quantum, quantum_cost, n, k, 'Quantum', nodeMap)
    #x_quantum_2d = get_traversed_path(x_quantum,n)
    qubit_needed = nodes_num * (nodes_num-1)
    new_xc, new_yc = get_new_coord(xc,yc, result, nodes_num)
    return new_xc, new_yc, result, solution.total_cost(), nodeMap, qubit_needed


def admm(n,k,nodes):
    print('admm implementation')

def classical(n,k):
    print('**********************classical implementation**********************')
    xc, yc, instance, nodeMap = getRandomNodesFromDb(n)
    # Solve the problem in a classical fashion via CPLEX
    x = None
    z = None
    x,z,classical_cost = get_classical_solution(n,k,instance)
    x_classical = np.zeros(n**2)
    kk = 0
    for ii in range(n ** 2):
        if ii // n != ii % n:
            x_classical[ii] = z[kk]
            kk +=  1

    print(x_classical)
    # if x is not None:
    #     visualize_solution(xc, yc, x, classical_cost, n, k, 'Classical', nodeMap)
    x_classical_2d = get_traversed_path(x_classical,n)
    solution = []
    solution.append(x_classical_2d)
    qubit_needed = 0
    new_xc, new_yc = get_new_coord(xc,yc, solution, n)
    print(new_xc)
    print(new_yc)
    print(classical_cost)
    return new_xc, new_yc, x_classical_2d, classical_cost, nodeMap, qubit_needed

def get_classical_solution(n,k,instance):
    x = None
    z = None
    classical_cost = None
    try:
        x,classical_cost = cplex_solution(n,k,instance)

        # Put the solution in the z variable
        z = [x[ii] for ii in range(n**2) if ii//n != ii%n]
        
        # Print the solution
        print('z:',z)
        print('x:',x)
    except:
        print("CPLEX may be missing.")
    return x,z,classical_cost
#classical(4,1)
#qaoa(4,1)
#vqe(4,1)
# getRoute(3,1,'classical')