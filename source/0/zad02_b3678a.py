# https://github.com/Denwiliusz27/cryptography/blob/277a6744d9b276dffb59b93d44a8ca49988af3bb/cw10/zad02.py
from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.visualization import plot_histogram
import numpy as np


if __name__ == '__main__':
    qc = QuantumCircuit(3, 3)

    # 001
    # qc.initialize('1', 2)

    # 010
    qc.initialize('1', 1)

    qc.u(np.pi / 3, np.pi, np.pi, 0)  # theta=pi/3,phi=pi,lambda=pi
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 2)

    qc.measure([0, 1, 2], [0, 1, 2])

    # qc.draw(output='mpl', filename='zad02.png')

    backend = BasicAer.get_backend('qasm_simulator')
    counts = execute(qc, backend, shots=1000).result().get_counts()

    for key in counts:
        counts[key] = counts[key]/1000

    print(counts)
    plot_histogram(counts, filename='zad02_010.png')