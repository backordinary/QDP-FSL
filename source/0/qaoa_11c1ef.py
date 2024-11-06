# https://github.com/dlasecki/QaoaParamsPredictor/blob/250b48d36d033adfe3d02a59e3e4a34991834e8f/predictor/qaoa_solver/qaoa.py
from math import pi

from numpy import concatenate
from numpy.random.mtrand import uniform
from qiskit import Aer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization.applications.ising.common import sample_most_likely

from data_structures.problem_instances.graph_problems.graph_problem_instance import ProblemInstance


def qaoa_with_optimizer(problem_instance: ProblemInstance):
    """Runs the QAOA algorithm with a classical optimizer of hyperparameters and returns a ProblemInstance containing
    a solution."""
    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend)

    min_cost = None
    result_min_cost = None
    all_results = []
    for _ in range(problem_instance.num_starting_points):
        result = qaoa(quantum_instance, problem_instance,
                      initial_point=_generate_uniformly_random_parameters(problem_instance.p),
                      optimizer=problem_instance.optimizer.optimizer)
        qaoa_expectation = result['eigenvalue'].real + problem_instance.offset
        all_results.append(result)
        if _is_solution_better(min_cost, qaoa_expectation):
            min_cost, result_min_cost = qaoa_expectation, result

    problem_instance.good_params = _get_high_quality_solutions(all_results, result_min_cost, problem_instance.offset)
    most_likely_binary_solution = sample_most_likely(result_min_cost['eigenstate'])
    most_likely_solution_value = problem_instance.calc_objective_value(most_likely_binary_solution)
    problem_instance = _get_instance_with_best_solution(problem_instance, -min_cost,
                                                        result_min_cost['optimal_point'],
                                                        most_likely_binary_solution,
                                                        most_likely_solution_value)

    return problem_instance


def qaoa(quantum_instance, problem_instance, initial_point, optimizer=None):
    """Runs the QAOA algorithm without a classical optimizer."""
    aqua_globals.massive = True
    qubit_operator = problem_instance.qubit_operator
    p = problem_instance.p

    qaoa_object = QAOA(operator=qubit_operator, p=p, initial_point=initial_point, optimizer=optimizer)
    return qaoa_object.run(quantum_instance)


def _generate_uniformly_random_parameters(p: int):
    """Generates random parameters that are then used as initial parameters for a QAOA algorithm."""
    return concatenate([uniform(low=0.0, high=2 * pi, size=p), uniform(low=0.0, high=pi, size=p)])


def _is_solution_better(min_cost, qaoa_expectation):
    """Checks if a solution provided is better than a currently known best solution provided."""
    return not min_cost or qaoa_expectation < min_cost


def _get_instance_with_best_solution(problem_instance: ProblemInstance, min_cost, optimal_params,
                                     most_likely_binary_solution, most_likely_solution_value):
    """Fills a problem instance provided with optimal solution data provided."""
    problem_instance.optimal_value = min_cost
    problem_instance.optimal_params = optimal_params
    problem_instance.most_likely_binary_solution = most_likely_binary_solution
    problem_instance.most_likely_solution_value = most_likely_solution_value

    return problem_instance


def _get_high_quality_solutions(all_results, best_result, offset):
    """Filters solutions obtained from several runs to these with low approximation error."""
    high_quality_solutions_params = []
    best_objective_value = best_result['eigenvalue'].real + offset
    allowed_approximation_error = 0.01  # 1% approximation error
    for solution in all_results:
        solution_objective_value = solution['eigenvalue'].real + offset
        optimality_ratio = solution_objective_value / best_objective_value
        approximation_error = 1 - optimality_ratio
        if allowed_approximation_error >= approximation_error >= 0.0:
            high_quality_solutions_params.append(solution['optimal_point'])

    return high_quality_solutions_params
