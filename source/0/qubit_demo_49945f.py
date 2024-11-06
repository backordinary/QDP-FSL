# https://github.com/mariuskriukovas/Projektinis/blob/ed4de5f13a6053a9ef1f18e133b6fc70fca3bc60/experiments/qubit_demo.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram

import git.Projektinis.tools.functions as fun
import git.Projektinis.tools.gates as my_gt
import git.Projektinis.tools.simulators as tools

test_matrix = np.array([[0.58168309 + 0.j, -0.8134155j], [-0.8134155j, 0.58168309 + 0.j]])


def math_simulation(test_matrix):
    zero = my_gt.get_zero_ket(2)
    fun.to_latex(test_matrix)
    # print(fun.is_unitary(test_matrix))
    v = fun.mul(test_matrix, zero)
    fun.to_latex(v)
    v_ermitian = fun.calc_bra(v)
    fun.to_latex(v_ermitian)
    length = fun.mul(v_ermitian, v)
    print(length)
    fun.find_prob(v)


math_simulation(test_matrix)


def quiskit_simuliation(test_matrix):
    qr = QuantumRegister(1, "qr")
    cr = ClassicalRegister(1, "cr")

    qc = QuantumCircuit(qr, cr)

    qc.unitary(test_matrix, [0], label='$ U_{perėjimo} $')
    qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];

    qc.draw(output='mpl')
    result = tools.simulate(qc).get_counts()
    # plt.show()
    # result = tools.simulate_unitary(qc).get_unitary(qc, decimals=3) #.get_counts()
    # print(result)
    plot_histogram(result)
    plt.show()


quiskit_simuliation(test_matrix)
