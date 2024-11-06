# https://github.com/leonardompp/pfcf_codes/blob/6698a4c1cbb9d8fa3c59a7781b081bfc716a516c/src/test_script.py
"""
This is a temporary script that shall be deleted later
It serves to check if all packages were installed and to generate the requirements file
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile

from qiskit.tools.visualization import plot_histogram

import matplotlib.pyplot as plt
import numpy as np

def main():
    # Simple bell state creation
    qc = QuantumCircuit(QuantumRegister(1, "q0"), QuantumRegister(1, "q1"), ClassicalRegister(2))

    # See which state will be created
    i = np.random.randint(0, 2) # Qubit 1
    j = np.random.randint(0, 2) # Qubit 0

    # Circuit and measure
    if i == 1:
        qc.x(1)
    if j == 1:
        qc.x(0)

    qc.h(1)
    qc.cx(1, 0)
    qc.measure([0, 1], [0, 1])

    # Plot in mpl
    qc.draw(output='mpl', initial_state=True, reverse_bits=True)  # Reverse for visualization, highest is MSQ
    plt.show()

    # Compile and run locally
    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, simulator)

    result = simulator.run(transpiled).result()
    data = result.get_counts()
    print(data)
    plot_histogram(data, title=f"Measurements for Bell State $|\psi-{i}{j}>$")
    plt.show()

if __name__ == "__main__":
    main()