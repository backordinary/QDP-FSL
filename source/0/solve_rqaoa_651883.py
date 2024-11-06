# https://github.com/jedwvv/QAOAvsVQE/blob/624a40112d4cdf919e5be752831d11786febf389/solve_rqaoa.py
from time import time
import warnings
import multiprocessing as mp
import numpy as np
import pickle as pkl
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.quantum_info import Statevector
import QAOAEx
from parser_all import parse
from generate_qubos import solve_classically, arr_to_str, str_to_arr, visualise_solution
from generate_problem import import_map
from classical_optimizers import NLOPT_Optimizer
from qiskit_optimization import QuadraticProgram
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from solve_qubo_qaoa import get_costs, Fourier_QAOA, n_qbit_mixer, construct_initial_state

warnings.filterwarnings('ignore')

def main(args=None):
    start = time()
    if args == None:
        args = parse()

    prob_s_s = []
    qubo_no = args["no_samples"]
    print("__"*50, "\nQUBO NO: {}\n".format(qubo_no), "__"*50)

    #Load generated qubo_no
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], qubo_no), 'rb') as f:
        qubo, max_coeff, operator, offset, routes = pkl.load(f)
    # print(operator)
    classical_result = solve_classically(qubo)
    print(classical_result)
    x_arr = classical_result.x
    optimal_value = qubo.objective.evaluate(x_arr)
    # print(optimal_value)
    x_str = arr_to_str(x_arr)
    sort_values = get_costs(qubo)
    # print("_"*50)
    # up_to = 27
    # print("{} lowest states:".format(up_to))
    # avg = 0
    # for i in range(up_to):
    #     print(sort_values[i])
    #     avg += sort_values[i][1]
    # print("_"*50)
    # print("Avg: {}".format(avg/up_to))


    #Remake QUBO to introduce only ZiZj terms
    op, offset = qubo.to_ising()
    new_operator = []
    for i, op_1 in enumerate(op):
        coeff, op_1 = op_1.to_pauli_op().coeff, op_1.to_pauli_op().primitive
        op_1_str = op_1.to_label()
        Z_counts = op_1_str.count('Z')
        if Z_counts == 1:
            op_1_str = 'Z' + op_1_str
        else:
            op_1_str = 'I' + op_1_str
        pauli = PauliOp( primitive = Pauli(op_1_str), coeff = coeff )
        new_operator.append(pauli)
    qubo_2 = QuadraticProgram()
    operator = sum(new_operator)
    qubo_2.from_ising(operator, offset, linear=True)

    print("Solving with QAOA...")
    no_shots = 10000
    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend, shots = no_shots)
    optimizer_method = "LN_SBPLX"
    optimizer = NLOPT_Optimizer(method = optimizer_method)
    print("_"*50,"\n"+optimizer.__class__.__name__)
    print("_"*50)


    p = 1
    delta_points = []
    deltas = np.arange(0.75, 0.76, 0.05)
    print("delta_t's: ", deltas)
    original_point = np.append([], [[(i+1)/p, 1-(i+1)/p] for i in range(p)])
    for delta in deltas:
        delta_points.append(delta*original_point)
    
    # point = [ 0.60081404,  0.11785113,  0.02330747,  1.10006101,  0.46256391,
    #    -0.96823671]
    draw_circuit = True
    initial_state = construct_initial_state_2(args["no_routes"], args["no_cars"])
    mixer = n_qbit_mixer(initial_state)
    quantum_instance = QuantumInstance(backend)
    results = []
    exp_vals = []
    for point in delta_points:
        point = QAOAEx.convert_to_fourier_point(point, 2*p)
        qaoa_results, _ = Fourier_QAOA(operator,
                                                    quantum_instance,
                                                    optimizer,
                                                    reps = p,
                                                    initial_fourier_point=point,
                                                    initial_state = initial_state,
                                                    mixer = mixer,
                                                    construct_circ=draw_circuit,
                                                    fourier_parametrise = True
                                                    )
        # print(optimal_circ.draw())
        results.append(qaoa_results)
        exp_val = qaoa_results.eigenvalue + offset
        print("ExpVal {}".format(exp_val))
        exp_vals.append(exp_val)
    minim_index = np.argmin(exp_vals)
    qaoa_results = results[minim_index]
    sort_states = sorted(qaoa_results.eigenstate.items(), key=lambda x: x[1], reverse=True)
    correlations = get_correlations(sort_states)
    # print(correlations)
    i,j = find_strongest_correlation(correlations)
    if correlations[i,j] > 0:
        condition = "Z_{i} = Z_{j}".format(i=i,j=j)
    else:
        condition = "Z_{i} = - Z_{j}".format(i=i,j=j)
    print("Max: <Z_{i}Z_{j}>={maxim}, condition: ".format(i=i, j=j, maxim = correlations[i,j])+condition)
    ground_energy = sort_values[0][1]
    x_s = [x_str]
    if sort_values[0][0] == x_str or np.round(sort_values[0][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[0][0]))
        x_s.append(sort_values[0][0])
    if sort_values[1][0] == x_str or np.round(sort_values[1][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[1][0]))
        x_s.append(sort_values[1][0])
    if sort_values[2][0] == x_str or np.round(sort_values[2][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[2][0]))
        x_s.append(sort_values[2][0])
    if sort_values[3][0] == x_str or np.round(sort_values[3][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[3][0]))
        x_s.append(sort_values[3][0])
    if sort_values[4][0] == x_str or np.round(sort_values[4][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[4][0]))
        x_s.append(sort_values[4][0])
    if sort_values[5][0] == x_str or np.round(sort_values[5][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[5][0]))
        x_s.append(sort_values[5][0])
    if sort_values[6][0] == x_str or np.round(sort_values[6][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[6][0]))
        x_s.append(sort_values[6][0])
    if sort_values[7][0] == x_str or np.round(sort_values[7][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[7][0]))
        x_s.append(sort_values[7][0])
    if sort_values[8][0] == x_str or np.round(sort_values[8][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[8][0]))
        x_s.append(sort_values[8][0])
    if sort_values[9][0] == x_str or np.round(sort_values[9][1], 4) != np.round(ground_energy, 4):
        continue
    else:
        print("Other ground state(s) found: '{}'".format(sort_values[9][0]))
        x_s.append(sort_values[9][0])
    x_s_2 = []
    print("All the possible solutions were originally: ")
    print("[Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9]")
    for string in x_s:
        x = '0' + string
        x_arr = np.array(str_to_arr(x))
        # print("f({x})={obj}".format(x=x, obj = qubo_2.objective.evaluate(x_arr)))
        x_s_2.append(x_arr)
        formatted_arr = ["{} ".format(item) for item in x_arr]
        formatted_arr = ' '.join(formatted_arr)
        print("[{}]".format(formatted_arr))


    finish = time()
    print("Time taken {time}s".format(time=finish-start))

def construct_initial_state_2(no_routes: int, no_cars: int):
    """[summary]

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    R = no_routes
    N = no_cars
    w_one = QuantumCircuit(R)
    for r in range(0,R-1):
        if r == 0:
            w_one.ry(2*np.arccos(1/np.sqrt(R-r)),r)
        elif r != R-1:
            w_one.cry(2*np.arccos(1/np.sqrt(R-r)), r-1, r)
    for r in range(1,R):
        w_one.cx(R-r-1,R-r)
    w_one.x(0)

    #Tensor product for N cars
    initial_state = w_one.copy()
    for _ in range(0,N-1):
        initial_state = initial_state.tensor(w_one)
    anc = QuantumRegister(1)
    initial_state.add_register(anc)

    initial_state.name = "U_init"

    return initial_state

def get_correlations(state) -> np.ndarray:
        """
        Get <Zi x Zj> correlation matrix from the eigenstate(state: Dict).

        Returns:
            A correlation matrix.
        """
        states = state
        # print(states)
        x, prob = states[0]
        n = len(x)
        correlations = np.zeros((n, n))
        for x, prob in states:
            for i in range(n):
                for j in range(i):
                    if x[n-i-1] == x[n-j-1]:
                        correlations[i, j] += prob
                    else:
                        correlations[i, j] -= prob
        return correlations

def find_strongest_correlation(correlations):

        # get absolute values and set diagonal to -1 to make sure maximum is always on off-diagonal
        abs_correlations = np.abs(correlations)
        for i in range(len(correlations)):
            abs_correlations[i, i] = -1

        # get index of maximum (by construction on off-diagonal)
        m_max = np.argmax(abs_correlations.flatten())

        # translate back to indices
        i = int(m_max // len(correlations))
        j = int(m_max - i*len(correlations))
        return (i, j)

main()