# https://github.com/lasys/hm_master_thesis/blob/9e605895705534bc0505f19240152c0ebaaf2a30/benchmark/noise_ibm/helpers/recursive_ws_helper.py
from qiskit_optimization.algorithms import (
    WarmStartQAOAOptimizer,
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    GoemansWilliamsonOptimizer,
    WarmStartQAOAFactory,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
from .qaoa_helper import *

class MyWarmStartQAOAFactory(WarmStartQAOAFactory):
    def create_mixer(self, initial_variables: List[float]) -> QuantumCircuit:
        """
        Creates an evolved mixer circuit as Ry(theta)Rz(-2beta)Ry(-theta).
        Args:
            initial_variables: Already created initial variables.
        Returns:
            A quantum circuit to be used as a mixer in QAOA.
        """
        circuit = QuantumCircuit(len(initial_variables))
        beta = Parameter("beta")

        for index, relaxed_value in enumerate(initial_variables):
            theta = 2 * np.arcsin(np.sqrt(relaxed_value))
            
            circuit.ry(theta, index)
            circuit.rz(-2.0*beta, index)
            circuit.ry(-theta, index)

        return circuit



def _run_recursive_ws_qaoa(max_cut, qaoa, epsilon=0.25):
    ws_qaoa = WarmStartQAOAOptimizer(pre_solver=GoemansWilliamsonOptimizer(5),
                                     num_initial_solutions=5, warm_start_factory=MyWarmStartQAOAFactory(epsilon),
                                     relax_for_pre_solver=False, qaoa=qaoa, epsilon=epsilon)
    optimizer = RecursiveMinimumEigenOptimizer(ws_qaoa)
    result = optimizer.solve(max_cut.to_qubo())
    optimal_parameters = qaoa.optimal_params
    
    return result, optimal_parameters


def run_recursive_ws_qaoa(max_cut, qaoa, epsilon=0.25, print_output=False):
    result, optimal_parameters = _run_recursive_ws_qaoa(max_cut, qaoa, epsilon)
    
    mean, distribution = max_cut.analyse(result, print_output=print_output)
    if print_output:
        print(f"Optimal Parameters: {optimal_parameters % 3.14}")
        print(f"Run Recursive WarmStartQAOAOptimizer with epsilon: {epsilon}")
        max_cut.plot_histogram(distribution, mean)
        
    return result, mean, optimal_parameters

def _run_evaluation_recursive_ws_qaoa(max_cut, qaoa, epsilon=0.25, print_output=False):
    result, optimal_parameters = _run_recursive_ws_qaoa(max_cut, qaoa, epsilon)
    mean, r, ar = max_cut.analyse_evaluation(result, print_output=print_output)
    return mean, r, ar


def start_recursive_ws_qaoa_evaluation(max_cut, eval_num, reps, epsilon, maxiter=50):
    
    means = []
    ratios = []
    approx_ratios = []
    print(f"p={reps}: ",end='')
    for i in range(0, eval_num):
        qaoa = create_ws_qaoa(reps=reps, optimizer=COBYLA(maxiter=maxiter))
        try:
            mean,r,ar = _run_evaluation_recursive_ws_qaoa(max_cut, qaoa=qaoa, epsilon=epsilon)
            means.append(mean)
            ratios.append(r)
            approx_ratios.append(ar)
        except Exception as e:
            print(e)
            try:
                mean,r,ar = _run_evaluation_recursive_ws_qaoa(max_cut, qaoa=qaoa, epsilon=epsilon)
                means.append(mean)
                ratios.append(r)
                approx_ratios.append(ar)
            except:
                print(f"Cannot run evaluation {i} with p={reps}")
                
        
        print(f".",end='')
    print()
    
    return means, ratios, approx_ratios

        