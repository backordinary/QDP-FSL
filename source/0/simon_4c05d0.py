# https://github.com/klam98/CMPT409-group-project/blob/eae1b801385df127a62398df0026e930b41ee1a6/group%20project/simon.py
# Adapted from: https://qiskit.org/textbook/ch-algorithms/simon.html

# ignore deprecation warnings
import logging
import warnings
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, transpile, assemble

# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import simon_oracle

from black_box import BlackBox


def test_simon(bitstring):
    """Run experiment on bitstring."""
    print("Setting bitstring as %s..." % bitstring)
    n = len(bitstring)

    print("Generating Simon circuit using 3 qubits...")
    simon_circuit = QuantumCircuit(n*2, n)

    # Apply Hadamard gates before querying the oracle
    simon_circuit.h(range(n))

    # Apply barrier for visual separation
    simon_circuit.barrier()

    # creates a Simon oracle for the bitstring b
    simon_circuit += simon_oracle(bitstring)

    # Apply barrier for visual separation
    simon_circuit.barrier()

    print("Applying Hadamard gates...\n")
    # Apply Hadamard gates to the input register
    simon_circuit.h(range(n))

    # Measure qubits
    simon_circuit.measure(range(n), range(n))
    print(simon_circuit)
    print("DONE.\n")


    ## Experiment using local simulator
    print("Generating a random distribution of qubit states using the previously generated Simon circuit...")
    aer_sim = Aer.get_backend('aer_simulator')
    shots = 1024
    qobj = assemble(simon_circuit, shots=shots)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()

    for key in counts.keys():
        counts[key] = str(counts[key] / 10) + "%"

    print("Probability distribution of qubit states: %s\n" % counts)

    # test each input to see if the black box is one-to-one or two-to-one
    bb = BlackBox(bitstring)
    print("Black box 1:")
    bb1_res, bb1_time = bb.run(counts, BlackBox.box1)
    print("Black box 2:")
    bb2_res, bb2_time = bb.run(counts, BlackBox.box2)
    return {
        "bb1_res": bb1_res,
        "bb1_time": bb1_time,
        "bb2_res": bb2_res,
        "bb2_time": bb2_time,
    }


if __name__ == "__main__":
    bitstring = '110'
    test_simon(bitstring)
