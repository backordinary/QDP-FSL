# https://github.com/simondobers/quantum_alternating_operator/blob/741502232df57f3b2965505c498937cebaf2a3a9/qaoa.py
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import SummedOp,PrimitiveOp,I
from qiskit.circuit import Parameter
from qiskit import Aer,transpile,QuantumCircuit
from typing import Dict, List,Callable,Tuple
import numpy as np
from alternating_operator import create_initial_state_circuit
from helper import bitstring_to_path,cost
import pickle

# returns the bit index for an alpha and j
def bit(ncities,alpha, j):
    return j * ncities + alpha

def D(ncities,alpha, j):
    qubit = bit(ncities,alpha, j)
    nqubits = ncities**2

    pauli_string = ''.join(['I' if i!=qubit else 'Z' for i in range(nqubits-1,-1,-1) ])

    return 0.5 * PrimitiveOp(Pauli(''.join(['I' for _ in range(nqubits-1,-1,-1) ]))) - PrimitiveOp(Pauli(pauli_string))

"""
weights and connections are numpy matrixes that define a weighted and unweighted graph
penalty: how much to penalize longer routes (penalty=0 pays no attention to weight matrix)
"""
def build_cost(penalty, ncities, G):
    nqubits = ncities**2
    identity_string = ''.join(['I' for _ in range(nqubits-1,-1,-1) ])

    ret = PrimitiveOp(Pauli(identity_string))
    # constraint (a)
    for i in range(ncities):
        cur = PrimitiveOp(Pauli(identity_string))
        for j in range(ncities):
            cur -= D(ncities,i, j)
        ret += cur.compose(cur)**2

    # constraint (b)
    for i in range(ncities):
        cur = PrimitiveOp(Pauli(identity_string))
        for j in range(ncities):
            cur -= D(ncities,j, i)
        ret += cur**2

    # constraint (d) (the weighting)
    for i in range(ncities-1):
        cur = PrimitiveOp(Pauli(identity_string))
        for j in range(ncities):
            for k in range(ncities):
                if not np.isclose(G[j, k],0.0):
                    cur -= D(ncities,j, i).compose(D(ncities,k, i+1),front=True) * G[j, k]
        ret += cur * penalty
        
    return ret.reduce()

def create_classical_qaoa_circ(G:np.array,penalty=1.,reps=1) -> QuantumCircuit:
    """Creates the qaoa circuit including the constraints for the TSP Problem 

    Args:
        G (np.array): TSP graph, represented as symmetric np.array
        reps (int): Nr. of repetions of mixer-cost_ham steps. 
        penalty (float, optional): Parameter how much to penalize invlaid paths. Defaults to 1..

    Returns:
        qiskit.QuantumCircuit: Ansatz circuit including measurements
    """
    ncities = G.shape[0]

    qaoa = QAOA(optimizer=COBYLA(),reps=reps,initial_state =create_initial_state_circuit(ncities),quantum_instance=Aer.get_backend('aer_simulator'))

    cost_ham = build_cost(penalty,ncities,G).reduce()
    params = [Parameter(str(i)) for i in range(reps*2)]
    qc = qaoa.construct_circuit(params,cost_ham)[0]

    qc.measure_all()

    return qc

def compute_expectation(counts:Dict, G:np.array, print_progress=True)->float:
    """Computes the expectation of the cost for a given simulation result.

    Args:
        counts (Dict): Key value pairs of bitsting(QuantumState) and how often this state was measured. E.g. {'100010001':317,'100001010':210}
        G (np.array): TSP graph, represented as symmetric np.array

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
           # tbd how to penalize this case 
           # as if one would travel all paths with maximal cost 
           path_cost = G.max() * G.shape[0]
            
        avg += path_cost * count
        sum_count += count
    if print_progress:
        print(f"Current expected cost: {round(avg/sum_count,2)}")  
          
    return avg/sum_count 


def get_expectation_qaoa(G:np.array, reps:int, shots=512,penalty=1.,log_intermediate_counts=False)-> Callable[[List[float]],float]:
    """ Returns function that takes as argument a list of parameters [ß,y] and computes the expectation for that parametrization

    Args:
        G (np.array): TSP graph, represented as symmetric np.array
        reps (int): Nr. of Troterrization steps. 
        shots (int, optional): No. of shots for the circuit. Defaults to 512.
        penalty (float, optional): Parameter how much to penalize invlaid paths. Defaults to 1..

    Returns:
        Callable[[List[float]],float]: Function that takes as argument a list of parameters [ß,y] and computes the expectation(cost) for that parametrization.
                                       This function gets passed to the optimizer in order to find the best parameters for the circuit.
    """
    simulator = Aer.get_backend('aer_simulator')
    simulator.shots = shots    

    qc = create_classical_qaoa_circ(G, reps=reps, penalty=penalty)
    qc = transpile(qc, simulator,optimization_level = 1)

    # initialize saving of data
    if log_intermediate_counts:
        with open(".\\data\\qaoa_counts", "wb") as fp:   #Pickling
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
            with open(".\\data\\qaoa_counts", "rb") as fp:   # Unpickling
                        qaoa_counts = pickle.load(fp)
                        qaoa_counts.append(counts)

            with open(".\\data\\qaoa_counts", "wb") as fp:   #Pickling
                pickle.dump(qaoa_counts,fp)

        return compute_expectation(counts, G)
    
    return execute_circ