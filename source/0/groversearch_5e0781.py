# https://github.com/goobta/ma3233-final-project/blob/ed3263a4ef348de77cf97109711c364947aeed5f/graph_utils/groversearch.py
from qiskit import BasicAer
from qiskit.tools.visualization import plot_histogram
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import TruthTableOracle

from typing import Dict, Tuple, List
from time import time

#from evaluators import HamiltonianEvaluator
#import graph_types as t
from . import graph_types as t
from . evaluators import HamiltonianEvaluator


def call_grover(truth_map: str, num_vertices: int, shots=1024) -> dict:
    """Call the simulation for grover's algorithm with the truth map and time its execution

    :param truth_map: The string bitmap
    :param num_vertices: Number of vertices of the graph for documentation purposes
    :return: the GroverResult item
    """
    start = time()

    oracle = TruthTableOracle(truth_map)
    grover = Grover(oracle)  # Wow that's nice that this already exists
    result = grover.run(QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=shots))

    end = time()
    print('Grover\'s search on n = {} vertices:\nTime elapsed: {}s\n'.format(num_vertices, end - start))
    return result


def get_truth_map(truth_table: Dict[Tuple[t.edge, ...], bool]) -> str:
    """Take in a dictionary truth mapping and convert it to the readable bitmap

    :param truth_table: The tuple:bool dictionary
    :return: A string of the equivalent truth_table
    """
    # Oracle takes a string representation of the binary truth table like '1100001'
    binary_repr = ''
    for key, val in truth_table.items():
        binary_repr += str(int(val))  # Abuse of casting to send True -> 1 -> '1'

    # Length of bitmap needs to be a power of 2
    # This bitwise operation checks to see that the value of length is in fact a power of 2
    # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
    # Add '0' to the END, the bitmap is considered ordered with the MSB as the 0th bit.
    while not (len(binary_repr) != 0 and len(binary_repr) & (len(binary_repr) - 1) == 0):
        binary_repr += '0'

    return binary_repr


def run_grovers(graph_edge_set: List[t.edge], plot: bool = False, graph_str: str = None) -> List[t.edge]:
    """Run Grover's Algorithm on the given graph edge set

    :param graph_edge_set:
        The set of edges for the graph. Comes in a list of Tuple[u, v] form.
            Tuple[u,v] is an edge in E where u and v are vertices in V.
    :param plot:
        True if you want it plotted by the visualizer.
    :param graph_str:
        An optional string representation of the graph to be printed.
    :return:
        A subset of the graph_edge_set that is a Hamilton cycle, or an empty list if one does not exist.
            Returns the subset in the same form as the edge set input.
    """
    if graph_str is not None:
        print(graph_str)

    evaluator = HamiltonianEvaluator(graph_edge_set)
    truth_table = evaluator.generate_truth_table()
    truth_map = get_truth_map(truth_table)

    if len(truth_map) <= 1:
        print('Unable to run Grover\'s search. Not enough edge combinations.\n')
        return []

    result = call_grover(truth_map, len(evaluator.vertices))

    if plot:
        plot_histogram(result['measurement'])  # One for Jupyter
        # plot_histogram(result['measurement']).savefig('visualizations/measurement{}.png'.format(truth_map))  # One
        # for me
        result['circuit'].draw(output='mpl', filename='visualizations/grover_circuit{}.png'.format(truth_map))
        print(result['circuit'].draw())

    # result[top_measurement] gives the binary number of the index of the found edge set
    index = int(result['top_measurement'], 2)
    result = list(truth_table.items())[index]

    # Check the result against the Oracle (Truth Table here) and return the edge set
    if result[1]:
        return list(result[0])
    else:
        return []
