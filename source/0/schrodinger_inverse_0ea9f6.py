# https://github.com/mariuskriukovas/Projektinis/blob/ed4de5f13a6053a9ef1f18e133b6fc70fca3bc60/experiments/schrodinger_inverse.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram

import git.Projektinis.experiments.schrodinger_with_Z_gates as previous_experiment
import git.Projektinis.tools.functions as fun
import git.Projektinis.tools.gates as my_gt
import git.Projektinis.tools.simulators as tools


def schrodinger_circle_inverse_gates(fi, measure=True):
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")

    qc = QuantumCircuit(qr, cr)

    qc.p(np.pi, qr[0])  # u1(pi) q[1];
    qc.h(qr[1])  # h q[2];
    qc.barrier()

    qc.cx(qr[1], qr[0])  # cx q[2],q[1]
    qc.barrier()

    qc.cx(qr[1], qr[0])  # cx q[2],q[1]
    qc.barrier()

    qc.p(np.pi, qr[0])  # u1(pi) q[1];
    qc.h(qr[1])  # h q[2];
    qc.barrier()

    # qc = qc + qc.inverse()

    if measure:
        qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];
        qc.measure(qr[1], cr[1])  # measure q[2] -> c[1];

    return qc


def schrodinger_inverse_gates_experiment():
    pi_val = np.pi / 2

    qc = schrodinger_circle_inverse_gates(pi_val, measure=True)

    qc.draw(output='mpl')
    # plt.show()
    counts = tools.simulate(qc).get_counts()
    print(counts)
    plot_histogram(counts, title="Grįžtamos sistemos būsenos")
    plt.show()


def math_experiment():
    tensor_I = fun.tensor_all([my_gt.H.get_value(), my_gt.p_gate.get_value(np.pi)])
    cx = previous_experiment.cx_gate()
    u_m1_m2 = fun.mul(tensor_I, cx)
    fun.to_latex(u_m1_m2)

    cx = previous_experiment.cx_gate()
    tensor_II = fun.tensor_all([my_gt.H.get_value(), my_gt.p_gate.get_value(np.pi)])
    u_2_1 = fun.mul(cx, tensor_II)
    fun.to_latex(u_2_1)

    u = fun.mul(u_m1_m2, u_2_1)
    fun.to_latex(u)


schrodinger_inverse_gates_experiment()
# math_experiment()
# print(fun.mul(my_gt.Z.get_value(), my_gt.Z.get_value()))
