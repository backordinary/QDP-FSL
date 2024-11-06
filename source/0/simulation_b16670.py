# https://github.com/talm95/QuantumComputing/blob/accf5574da5ecf8da7c233db928caf2ae74c9d4a/Utils/simulation.py
import matplotlib.pyplot as plt
from qiskit import Aer, transpile, assemble
from qiskit.visualization import plot_histogram


def simulate_with_aer(circuit, draw=True, scale=0.5, plot_results=True):
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(circuit, aer_sim)
    if draw:
        t_qc.draw(output='mpl', scale=scale)
        plt.show()
    qobj = assemble(t_qc)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()
    if plot_results:
        plot_histogram(counts)
        plt.show()
    return counts
