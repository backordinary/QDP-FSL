# https://github.com/BastienLaffon/workshop/blob/1a62515ef96738970119bd81f54a016855975fa1/src/scripts/helper.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def run_circuit(qc, simulator='statevector_simulator', shots=1, hist=True):
    # Tell Qiskit how to simulate our circuit
    backend = Aer.get_backend(simulator)

    # execute the qc
    results = execute(qc,backend, shots=shots).result().get_counts()

    print(results)
    # plot the results
    return results