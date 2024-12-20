# https://github.com/fjelljenta/Quantum-Challenge/blob/a8e66e65536bd8a6c9315e67d07778b8b96fb45f/quantimize/quantum_summary.py
import quantimize.classic.toolbox as toolbox
import quantimize.classic.classic_solution as csol
import quantimize.quantum.QGA as qga
import quantimize.quantum.quantum_neural_network as qna
import quantimize.quantum.quantum_solution as qsol
from qiskit import Aer


def quantum_genetic_algorithm_solution(flight_nr, dt=15, **kwargs):
    """Returns the quantum_genetic algorithm solution for a given flight and maps it to constant time points

    Args:
        flight_nr (int): Flight number

    Returns:
        dict: Trajectory in the format of a list of tuples (long, lat, time) embedded in a dict with flight number
    """
    run_complete = False
    while not run_complete:
        try:
            trajectory = qga.Q_GA(flight_nr, **kwargs)
            run_complete = True
        except:
            print("Retry QGA")
    if kwargs.get("timed_trajectory", True):
        trajectory = toolbox.timed_trajectory(trajectory, dt)
    return {"flight_nr": flight_nr, "trajectory": trajectory}


def quantum_neural_network(flight_nr, n_qubits):
    """Returns the quantum_neural_network algorithm solution for a given flight

    Args:
        flight_nr (int): Flight number
        n_qubits (int): number of qubits
        init_solution (array): initial solution from which the control points are extracted

    Returns:

    """
    report, solution, trajectory = csol.run_genetic_algorithm(flight_nr)
    print("Calculated boundary points")
    return qna.quantum_neural_network(flight_nr, n_qubits, solution['variable'])


def sample_grid():
    """returns the cost grid for a scaled-down experiment example (4 qubits)

    Returns:
        the cost grid
    """
    return qsol.sample_grid()


def run_QAOA(cg, orientation=0, verbose=False, runtime=False, backend=Aer.get_backend('qasm_simulator')):
    """Constructs and solves the mathematical function

    Args:
        cg: cost grid
        orientation: orientation of voxel_center_graph
        verbose (Bool): Output information or not

    Returns:
        z_sol: result in Pauli-Z form
        q_sol: result in qubit form
        vcg: voxel_center_graph

    """
    if runtime:
        return qsol.run_QAOA_real_backend(cg, orientation, verbose)
    return qsol.run_QAOA(cg, orientation, verbose, backend)


def compute_cost(trajectory):
    """Wrapper for the computation of the cost function

    Args:
        trajectory (list): List of trajectory points

    Returns:
        float: Cost of the flight
    """
    return toolbox.compute_cost(trajectory)

