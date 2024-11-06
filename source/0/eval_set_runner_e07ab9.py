# https://github.com/dlasecki/QaoaParamsPredictor/blob/970e41e3b633a1aa34b04a3daee9f0d4ad64ed9f/predictor/experiments/runners/eval_set_runner.py
import itertools
import multiprocessing
import pathlib
import time

from qiskit import Aer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.optimization.applications.ising.common import sample_most_likely

from data_structures.enums.problem_name import ProblemName
from data_structures.problem_instances.graph_problems.graph_problem_instance_factory import \
    create_graph_problem_instance
from data_structures.problem_instances.instances_generators.graph_problems.graphs_instances_generator import \
    generate_random_graph_instances, generate_ladder_graph_instances, generate_caveman_graph_instances, \
    generate_barbell_graph_instances
from experiments.data_handlers import results_serializer
from experiments.kde_model_provider import get_kde_model
from experiments.runners.base_runner import BaseRunner
from qaoa_solver import qaoa
from qaoa_solver.qaoa import _is_solution_better, _get_instance_with_best_solution

aqua_globals.massive = True


class EvaluationSetRunner(BaseRunner):
    """Used for running evaluation of ML models."""

    def worker(self, problem_name, input_graph, p_param, bandwidth, kernel):
        """Worker method for running QAOA algorithm (without a classical optimizer) for a given problem instance."""
        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend)
        problem_instance = create_graph_problem_instance(problem_name, p_param, input_graph)
        min_cost_expectation = None
        result_min_cost = None
        min_cost_params = None
        graph_type = input_graph.graph["graph_type"].value
        kde_model = get_kde_model(problem_name, graph_type, p_param, kernel, bandwidth)
        # we try 10 sampled points without an optimizer
        initial_points = kde_model.kde_model.sample(10)
        print(initial_points)
        for initial_point in initial_points:
            qaoa_res = qaoa.qaoa(quantum_instance, problem_instance, initial_point)
            qaoa_expectation = qaoa_res['eigenvalue'].real + problem_instance.offset
            if _is_solution_better(min_cost_expectation, qaoa_expectation):
                min_cost_expectation, result_min_cost, min_cost_params = qaoa_expectation, qaoa_res, initial_point
        most_likely_binary_solution = sample_most_likely(result_min_cost['eigenstate'])
        result_min_cost = _get_instance_with_best_solution(problem_instance, min_cost_expectation, min_cost_params,
                                                           most_likely_binary_solution, None)

        directory = self._build_model_validation_directory(problem_instance, problem_name)

        results_serializer.save_to_json(directory, result_min_cost, kernel, bandwidth)
        # print(result_min_cost.optimal_params, result_min_cost.optimal_value)
        print(result_min_cost.classical_solution_value)
        return result_min_cost.optimal_params, result_min_cost.optimal_value

    @staticmethod
    def _build_model_validation_directory(problem_instance, problem_name) -> pathlib.Path:
        """Builds a file directory depending on the problem instance and problem name provided where a result of the
        worker method can be stored."""
        path = pathlib.Path(__file__).parent.parent.resolve()
        path = (path).joinpath(pathlib.Path("kernel_density_estimation", "model_validation"))
        directory = pathlib.Path(problem_name.value,
                                 problem_instance.input_graph.graph["graph_type"].value + "_evaluation")
        full_path = (path).joinpath(directory)
        return full_path


if __name__ == '__main__':
    start = time.perf_counter()
    p_params = [1, 2, 3, 4]
    problems = [ProblemName.STABLE_SET]
    NUM_OF_PROCESSES = 8
    bandwidths = [0.2, 0.4, 0.6, 0.8]
    kernels = ["gaussian"]

    random_graph_num_of_vertices_test = [8, 12, 16, 20]
    random_graph_probabilities_test = [0.5, 0.6, 0.7, 0.8]

    ladder_graph_num_of_vertices_test = [2, 3, 5, 6, 7, 8, 9, 10, 11]

    caveman_graph_cliques_test = [(3, 4), (4, 4), (5, 4), (3, 3), (5, 3), (7, 3), (2, 3), (2, 5), (2, 6), (2, 7),
                                  (2, 8), (2, 9), (2, 10)]

    barbell_graph_num_of_vertices_test = [3, 5, 6, 7, 8, 9, 10, 11]

    eval_set_runner = EvaluationSetRunner()

    inputs_random = eval_set_runner._get_cartesian_product_of_inputs(problems, generate_random_graph_instances(
        random_graph_num_of_vertices_test, random_graph_probabilities_test),
                                                                     p_params, bandwidths, kernels)
    inputs_ladder = eval_set_runner._get_cartesian_product_of_inputs(problems, generate_ladder_graph_instances(
        ladder_graph_num_of_vertices_test),
                                                                     p_params, bandwidths, kernels)
    inputs_caveman = eval_set_runner._get_cartesian_product_of_inputs(problems, generate_caveman_graph_instances(
        caveman_graph_cliques_test),
                                                                      p_params, bandwidths, kernels)
    inputs_barbell = eval_set_runner._get_cartesian_product_of_inputs(problems, generate_barbell_graph_instances(
        barbell_graph_num_of_vertices_test),
                                                                      p_params, bandwidths, kernels)
    inputs = itertools.chain(inputs_barbell)

    with multiprocessing.Pool(processes=NUM_OF_PROCESSES) as pool:
        results = pool.starmap(eval_set_runner.worker, inputs)

    end = time.perf_counter()
    print(str(end - start) + " seconds")
