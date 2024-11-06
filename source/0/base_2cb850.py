# https://github.com/zaqwes8811/quantum/blob/7d9076dde96c2f852f7759c8196c8573cc1a50fa/base.py
# coding: utf-8

# import qiskit.tools.jupyter

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Circuit
    qc_output = QuantumCircuit(8)
    qc_output.measure_all()
    qc_output.draw(initial_state=True)
    plt.savefig("./circuit.png")

    # Simulate
    sim = Aer.get_backend('aer_simulator')
    result = sim.run(qc_output).result()
    counts = result.get_counts()
    plot_histogram(counts)
    plt.savefig("./histogram.png")
