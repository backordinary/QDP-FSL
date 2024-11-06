# https://github.com/andrea-ricciardi/qosf_project/blob/9f45d838b852196a105dab75386b3ec5554e6c1d/distributed_quantum_algorithms/parallel_montecarlo.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

from qae.applications.montecarlo.montecarlo_operators import SineSquaredOperator
from qae.applications.montecarlo.montecarlo_config import QAEMonteCarloConfig, IntegralConfig

from qiskit import Aer
from qiskit.algorithms.amplitude_estimators import EstimationProblem

from qae import UnitaryOperator
from parallelization import ParallelMaximumLikelihoodAmplitudeEstimation
from parallelization import ParallelQuantumInstance
from parallelization import GreedySchedule

from typing import List
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    """
    Run the MaximumLikelihoodAmplitudeEstimation parallely. Steps:

    (1) The user can change the inputs between 'Start Inputs' and 'End Inputs'.
    (2) Make the A & Q operators specific to the problem to estimate.
        Currently only the operator for the integral of (sin(x))^2 is implemented.
        In order to tackle new problems, the user must derive his/her own
        operator from operators.UnitaryOperator.
    (3) Set up the estimation problem through the EstimationProblem Qiskit class.
    (4) Make the schedule for parallel computing (resource allocation problem).
        The schedule does not allow for distributed computing, for that we will
        need to use the Interlin-q framework.
    (5) Set up the parallel MLAE algorithm.
    (6) Run the estimation problem.
    (7) Print the analytical result (in the case of the sine squared integral),
        the discretized result (equation (23) in Suzuki paper) and the
        estimation result, which estimates the discretized result.

    """
    ##############
    # (1) Inputs #
    ##############
    """ Start Inputs """
    # Number of qubits the operator A acts on ("n" in the Suzuki paper).
    # Notice that the measurement qubit is not included here.
    # By using more qubits, we can discretized (approximate) the integral better
    n_qubits_input = 3
    qubits_per_qpu = [10, 10, 10]
    problem = 'sine_squared'  # 'sine_squared' for computing the integral of (sin(x))^2
    evaluation_schedule = power_law_schedule()  # power_law_schedule() for power law, None for Suzuki
    """ End Inputs """

    # Make config for the program
    config = QAEMonteCarloConfig(n_qubits_input, qubits_per_qpu, problem,
                                 evaluation_schedule=evaluation_schedule)
    config.print_configuration()

    #######################
    # (2) Operators A & Q #
    #######################
    # Make the A and Q operators specific to the problem to estimate.
    # Currently only SineSquaredOperator is implemented.
    problem_operator = make_operator(config.integral, config.algo.n_qubits)
    state_preparation = problem_operator.prepare_state()  # P x R operator
    grover_operator = problem_operator.make_grover()  # Q operator

    #####################################
    # (3) Set up the estimation problem #
    #####################################
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=problem_operator.oracle_size - 1,  # -1 as it's the index of the qubit to measure
        grover_operator=grover_operator
    )

    ##########################################
    # (4) Schedule for distributed computing #
    ##########################################
    # A greedy schedule is a schedule that greedily fills the QPUs with as many qubits
    # that can possibly fit; when the QPUs cannot fit any more qubits, the execution
    # of those estimations are moved to the next round.
    # Other approaches are possible, even though not implemented, such as constraint programming.
    greedy_schedule = GreedySchedule(
        config.algo.num_circuits, config.hardware, problem_operator.oracle_size,
        allow_distributed=False
    )
    greedy_schedule.make_schedule()
    if len(greedy_schedule.schedule) == 0:
        raise SystemExit("Hardware capabilities are too small to handle the size of the oracle.")
    else:
        greedy_schedule.print_schedule()
    # greedy_schedule.schedule is a dictionary having:
    # - rounds as keys
    # - list of tuples (circuit_id, [qubits_qpu1, .., qubits_qpuN])

    ##########################################
    # (5) Set up the parallel MLAE algorithm #
    ##########################################
    # PMLAE needs, as well as an evaluation schedule, a parallelization schedule
    # and a ParallelQuantumInstance
    pmlae = ParallelMaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=config.algo.evaluation_schedule,
        parallelization_schedule=greedy_schedule,
        quantum_instance=ParallelQuantumInstance(Aer.get_backend('qasm_simulator'))
        )

    ##################################
    # (6) Run the estimation problem #
    ##################################
    result = pmlae.estimate(problem)

    #####################
    # (7) Print results #
    #####################
    print("### Results ###")
    print("Analytical result: {}".format(problem_operator.analytical_result()))
    print("Discretized result: {}".format(problem_operator.discretized_result()))
    print("Estimation result: {}".format(result.estimation))


def make_operator(integral_config: IntegralConfig, n_qubits: int) -> UnitaryOperator:
    """
    Make problem operator. Only SineSquaredOperator supported so far.

    Parameters
    ----------
    integral_config : IntegralConfig
    n_qubits : int

    Raises
    ------
    ValueError
        If integral_config.problem is not sine_squared.

    Returns
    -------
    Type[UnitaryOperator]
        Problem operator.

    """
    if integral_config.problem == 'sine_squared':
        return SineSquaredOperator(n_qubits, integral_config.param)
    else:
        raise ValueError("Operator for problem {} not implemented".format(
            integral_config.problem
        ))


def power_law_schedule(eps_precision: float = 0.01,
                       beta: float = 0.455) -> List[int]:
    """
    Power Law schedule

    Parameters
    ----------
    eps_precision : float, optional
        Epsilon precision. The default is 0.01.
    beta : float, optional
        Beta. The default is 0.455.

    Returns
    -------
    List[int]
        Power Law schedule.

    """
    max_k = max(int(1/eps_precision ** (2*beta)), int(math.log(1/eps_precision)))
    evaluation_schedule = [int(k ** ((1-beta)/(2*beta))) for k in range(1, max_k+1)]
    return evaluation_schedule


if __name__ == '__main__':
    main()
