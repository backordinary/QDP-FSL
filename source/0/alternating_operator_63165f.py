# https://github.com/simondobers/quantum_alternating_operator/blob/e17471b0259204f8f2c54def3859fc0f48de9ec0/alternating_operator.py
from typing import Dict, List,Callable,Tuple
from qiskit import QuantumCircuit,transpile,Aer
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit.library import QAOAAnsatz
from qiskit.opflow import SummedOp,PrimitiveOp
from qiskit.tools.visualization import plot_histogram
from helper import bitstring_to_path, cost
import numpy as np
import matplotlib.figure
import pickle
class InvalidMixerException(Exception):
    """Exception to raise if invalid solutions are measured (mixer has to preserve the feasible subspace)

    """
    pass


def create_initial_state_circuit(num_nodes : int) ->QuantumCircuit:
    """Returns a curuit that creates an initial state for the TSP problem, where at t=i city i is visited

    Args:
        num_nodes (int): number of cities in the TSP problem

    Returns:
        QuantumCircuit: Quantum circuit that creates the inital state
    """
    nqubits = num_nodes**2
    circ = QuantumCircuit(nqubits)

    # visit i'th city at time i
    for i in range(num_nodes):
        circ.x(i*num_nodes + i)

    return circ

def S_Plus(num_nodes : int, city : int, time:int)-> Operator:
    """Creates S_Plus matrix for {city} at {time} according to equation (56) from https://arxiv.org/pdf/1709.03489.pdf

    Args:
        num_nodes (int):  number of cities in the TSP problem
        city (int): affected city
        time (int): point in time where city is visited

    Returns:
        Operator: The S_Plus Operator
    """
    nqubits = num_nodes**2
    qubit = city * num_nodes + time
    
    # define Operator where X acts on qubit {qubit}, else I
    pauli_x_string = ''.join(['I' if i!=qubit else 'X' for i in range(nqubits-1,-1,-1) ])

    # define Operator where i*Y acts on qubit {qubit}, else I
    pauli_y_string = ''.join(['I' if i!=qubit else 'Y' for i in range(nqubits-1,-1,-1) ])

    return PrimitiveOp(Pauli(pauli_x_string)) + 1j* PrimitiveOp(Pauli(pauli_y_string))


def S_Minus(num_nodes : int, city : int, time:int)-> Operator:
    """Creates S_Minus matrix for {city} at {time} according to equation (57) from https://arxiv.org/pdf/1709.03489.pdf

    Args:
        num_nodes (int):  number of cities in the TSP problem
        city (int): affected city
        time (int): point in time where city is visited

    Returns:
        Operator: The S_Plus Operator
    """
    nqubits = num_nodes**2
    qubit = city * num_nodes + time
    
    # define Operator where X acts on qubit {qubit}, else I
    pauli_x_string = ''.join(['I' if i!=qubit else 'X' for i in range(nqubits-1,-1,-1) ])

    # define Operator where i*Y acts on qubit {qubit}, else I
    pauli_y_string = ''.join(['I' if i!=qubit else 'Y' for i in range(nqubits-1,-1,-1) ])

    return PrimitiveOp(Pauli(pauli_x_string)) - 1j* PrimitiveOp(Pauli(pauli_y_string))

def create_mixer_operator(num_nodes:int)->Operator:
    """Create mixing Operator according to equation (54)-(58) from https://arxiv.org/pdf/1709.03489.pdf

    Args:
        num_nodes (int): number of cities in the TSP problem

    Returns:
        Operator: Mixer Operator
    """
    mixer_operators = []
    for t in range(num_nodes-1):
        for city_1 in range(num_nodes):
            for city_2 in range(num_nodes):
                # swap city_1 at t=i, city_2 at t=i+1 <-> city_1 at t=i+1, city_2 at t=i
                # if current state is city_1 at t=i, city_2 at t=i+1 (see eq. (58))
                i = t
                u = city_1
                v = city_2
                first_part = S_Plus(num_nodes, u, i)
                first_part = first_part.compose(S_Plus(num_nodes, v, i+1),front=True)
                first_part = first_part.compose(S_Minus(num_nodes, u, i+1),front=True)
                first_part = first_part.compose(S_Minus(num_nodes, v, i),front=True)

                second_part = S_Minus(num_nodes, u, i)
                second_part = second_part.compose(S_Minus(num_nodes, v, i+1),front=True)
                second_part = second_part.compose(S_Plus(num_nodes, u, i+1),front=True)
                second_part = second_part.compose(S_Plus(num_nodes, v, i),front=True)
                mixer_operators.append((first_part + second_part))
    

    return SummedOp(mixer_operators)

def create_phase_separator(graph:np.array) -> Operator:
    """Create Phase Separator according to equation (53) from https://arxiv.org/pdf/1709.03489.pdf 

    Args:
        graph (np.array): TSP graph, represented as symmetric np.array

    Returns:
        Operator: Phase Separation Operator
    """
    phase_separators = []
    ncities = graph.shape[0]
    nqubits = ncities**2

    for t in range(ncities - 1):
        for city_1 in range(ncities):
            for city_2 in range(ncities):

                # If these aren't the same city 
                distance = graph[city_1, city_2]
                if city_1 != city_2 and distance != 0.0:
                    qubit_1 = t * ncities + city_1
                    qubit_2 = (t + 1) * ncities + city_2

                    pauli_z1_string = ''.join(['I' if i!=qubit_1 else 'Z' for i in range(nqubits-1,-1,-1) ])
                    pauli_z2_string = ''.join(['I' if i!=qubit_2 else 'Z' for i in range(nqubits-1,-1,-1) ])

                    # Append with Z1 * Z2 (compose needs reverse order)
                    phase_separators.append(distance * PrimitiveOp(Pauli(pauli_z2_string)).compose(PrimitiveOp(Pauli(pauli_z1_string))))
    
    return SummedOp(phase_separators)

def create_qaoa_circ(G:np.array,reps=1) -> QuantumCircuit:
    """Return the alternating operator ansatz circuit

    Args:
        G (np.array): TSP graph, represented as symmetric np.array
        reps (int, optional): Nr. of Troterrization steps. Defaults to 1.

    Returns:
        qiskit.QuantumCircuit: The circuit which implements the alternating operator ansatz, parametrized by reps*2 parameters
    """

    ncities = G.shape[0]
    nqubits = ncities**2
    
    inital_circ = create_initial_state_circuit(ncities)
    mixer = create_mixer_operator(ncities)
    phase_separator = create_phase_separator(G)
    ansatz = QAOAAnsatz(cost_operator=phase_separator, reps=reps, initial_state=inital_circ, mixer_operator=mixer, name='qaoaAnsatz')
    
    ansatz.measure_all()
    
    return ansatz

def get_expectation(G:np.array, reps:int, shots=512,log_intermediate_counts=False)-> Callable[[List[float]],float]:
    """ Returns function that takes as argument a list of parameters [ß,y] and computes the expectation for that parametrization

    Args:
        G (np.array): TSP graph, represented as symmetric np.array
        reps (int): Nr. of Troterrization steps. 
        shots (int, optional): No. of shots for the circuit. Defaults to 512.

    Returns:
        Callable[[List[float]],float]: Function that takes as argument a list of parameters [ß,y] and computes the expectation(cost) for that parametrization.
                                       This function gets passed to the optimizer in order to find the best parameters for the circuit.
    """
    simulator = Aer.get_backend('aer_simulator')
    simulator.shots = shots
    
    qc = create_qaoa_circ(G, reps=reps)
    qc = transpile(qc, simulator,optimization_level = 3)

    # initialize saving of data
    if log_intermediate_counts:
        with open(".\\data\\alternating_operator_counts", "wb") as fp:   #Pickling
            pickle.dump([], fp)


    def execute_circ(theta):
        # theta = [ß , y]

        # create parameter dictionary 
        params = {}
        for key,value in zip(qc.parameters,theta):
            params[key] = [value]

        counts = simulator.run(qc,parameter_binds=[params],  seed_simulator=10, 
                             nshots=shots).result().get_counts()
        
        if log_intermediate_counts:
            with open(".\\data\\alternating_operator_counts", "rb") as fp:   # Unpickling
                        alternating_operator_counts = pickle.load(fp)
                        alternating_operator_counts.append(counts)

            with open(".\\data\\alternating_operator_counts", "wb") as fp:   #Pickling
                pickle.dump(alternating_operator_counts,fp)


        return compute_expectation(counts, G)
    
    return execute_circ


def compute_expectation(counts:Dict, G:np.array, print_progress=True)->float:
    """Computes the expectation of the cost for a given simulation result.

    Args:
        counts (Dict): Key value pairs of bitsting(QuantumState) and how often this state was measured. E.g. {'100010001':317,'100001010':210}
        G (np.array): TSP graph, represented as symmetric np.array

    Raises:
        InvalidMixerException: If the mixer produces invalid states, i.e. does not preserve the feasible subspace.

    Returns:
        float: Expected cost.
    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        
        path = bitstring_to_path(bitstring)
        if path is not None:
            path_cost = cost(G, path)
        else:
            # this path should never be entered as the mixer should preserve the feasible subspace
            raise InvalidMixerException("Invalid mixer operator encountered. The current mixer produces infeasible results, check mixer operator.")
            
        avg += path_cost * count
        sum_count += count
    if print_progress:
        print(f"Current expected cost: {round(avg/sum_count,2)}")  
          
    return avg/sum_count 

def analyse_result(G:np.array,theta_res:List[float],reps=1,transform_labels_to_path=True,filter_unique_path=True,save_plot=False)->Tuple[matplotlib.figure.Figure,Dict]:
    """Creates a plot of the measurements of the qaoa circuit for a given parametrization

    Args:
        G (np.array):  TSP graph, represented as symmetric np.array
        theta_res (List[float]): list of parameters [ß,y] for the qaoa ciruit
        reps (int, optional): Nr. of Troterrization steps. . Defaults to 1.
        transform_labels_to_path (bool, optional): Wheter to transform the labels of the plot form bistring to a real path, see helper.bitstring_to_path . Defaults to True.

    Returns:
        Tuple[matplotlib.figure.Figure,Dict]: Histogram plot of counts, Key value pairs of bitsting(QuantumState) and how often this state was measured. E.g. {'100010001':317,'100001010':210}
    """
    simulator = Aer.get_backend('aer_simulator')
    simulator.shots = 1024
    
    qc = create_qaoa_circ(G,reps=reps)
    qc = transpile(qc, simulator,optimization_level=3)
    
    params = {}
    for key,value in zip(qc.parameters,theta_res):
        params[key] = [value]

    result = simulator.run(qc,parameter_binds=[params],  seed_simulator=10,nshots=1024).result()
    counts = result.get_counts()
    
    # remove duplicate path, e.g. [0,1,2] = [2,0,1]
    if filter_unique_path:
        counts = filter_unique_paths(G,counts)

    if save_plot:
        with open(".\\data\\alternating_operator_counts", "rb") as fp:   # Unpickling
                    alternating_operator_counts = pickle.load(fp)
                    alternating_operator_counts.append(counts)

        with open(".\\data\\alternating_operator_counts", "wb") as fp:   #Pickling
            pickle.dump(alternating_operator_counts,fp)


    fig = plot_histogram(counts, title='Result of Optimization')

    if transform_labels_to_path:
        ax = fig.axes
        ax = ax[0]
        
        labels = [bitstring_to_path(item.get_text(), return_as_string=True) for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
    
    return fig,counts
    

def filter_unique_paths(G:np.array,counts:dict, unique_cost_dict=None)->dict:
    """Removes all duplicate paths

    Args:
        counts (dict): _description_

    Returns:
        dict: _description_
    """

    if unique_cost_dict is None:
        cost_dict = {}
        for bitstring,count in counts.items():
            path = bitstring_to_path(bitstring)
            if path is not None :
                path_cost = round(cost(G,path),3)

                if path_cost not in cost_dict:
                    cost_dict[path_cost] = [bitstring,count]
                else:
                    cost_dict[path_cost] = [cost_dict[path_cost][0] , cost_dict[path_cost][1]+count]
            else :
                # invalid path, e.g. from qaoa 
                cost_dict[path_cost] = ['invalid',count]


    else : 
        cost_dict = {}
        unique_cost_dict = {round(val,3):key for key,val in unique_cost_dict.items()}

        for bitstring,count in counts.items():
            path = bitstring_to_path(bitstring)

            if path is not None :
                path_cost = round(cost(G,path),3)
              
                if path_cost not in cost_dict:
                    cost_dict[path_cost] = [unique_cost_dict[path_cost],count]
                else:
                    cost_dict[path_cost] = [unique_cost_dict[path_cost], cost_dict[path_cost][1]+count]

            else :
                # invalid path, e.g. from qaoa 
                cost_dict[np.inf] = [np.inf,count]

            cost_dict = dict(sorted(cost_dict.items(), key=lambda item: item[0]))

    return {value[0] : value[1] for _,value in cost_dict.items()}    
