# https://github.com/mariuskriukovas/Bakalaurinis/blob/f877821cb75d534b28c1d3d03ae2ac0df5cc68b6/tools/mock_circuit.py
# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# from qiskit.visualization import plot_histogram
#
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# import git.Bakalaurinis.tools.helper as tools
import matplotlib.pyplot as plt


def init_reg(n):
    qr = QuantumRegister(n, "qr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(qr, cr)
    q = (qr, cr, qc)
    return q


def init_b_gates(n, q):
    (qr, cr, qc) = q
    b_n = bin(n)[2:]
    print(b_n)
    for i, x in enumerate(reversed(b_n)):
        if x == '1':
            qc.x(qr[i])


def measure_all(q):
    (qr, cr, qc) = q
    for i, x in enumerate(qr):
        qc.measure(x, cr[i])


def get_MOCK_circuit():
    q = init_reg(5)
    init_b_gates(5, q)
    measure_all(q)
    return q

# from git.Bakalaurinis.simuliator.translator import QASMTranslator
#
# (qr, cr, qc) = get_MOCK_circuit()
# sim = QASMTranslator(qc.qasm()).get_simulator()
# print(sim.get_results())
# sim.show_circuit()
# plt.show()
# # def get_MOCK_results():
# #     (qr, cr, qc) = get_MOCK_circuit()
# #     return tools.simulate(qc)
