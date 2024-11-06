# https://github.com/talm95/QuantumComputing/blob/8e63961784806d863244ef3aab831af0f6bc668b/ShorsAlgorithm/QuantumPhaseEstimation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from QuantumFourierTransform import qft
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import pi


def qpe_rotations(qc, measure_qubits_num, unitary_matrix, psi_qubits):
    u_gate = unitary_matrix.to_gate()
    for qubit in range(measure_qubits_num):
        qc.h(qubit)
        u2n = QuantumCircuit(psi_n, name='U^' + str(2**(measure_qubits_num - qubit - 1)))
        for i in range(2**(measure_qubits_num - qubit - 1)):
            u2n.append(u_gate, range(psi_n))
        u2n_gate = u2n.to_gate()
        controlled_u2n = u2n_gate.control()
        qc.append(controlled_u2n,
                  list(np.append([qubit], list(range(measure_qubits_num, measure_qubits_num + psi_qubits)))))
    return qc


def qpe_qft_dagger(qc, measure_qubits_num):
    qf = QuantumCircuit(measure_qubits_num, name='QFT')
    qft(qf, measure_qubits_num)
    inv_qf = qf.inverse()
    qc.append(inv_qf, range(measure_qubits_num))
    return qc


def qpe_measure(qc, measure_qubits_num):
    for qubit in range(measure_qubits_num):
        qc.measure(qubit, qubit)
    return qc


def qpe(qc, measure_qubits_num, unitary_matrix, psi_qubits):
    qpe_rotations(qc, measure_qubits_num, unitary_matrix, psi_qubits)
    qpe_qft_dagger(qc, measure_qubits_num)
    qpe_measure(qc, measure_qubits_num)
    return qc


n = 3
psi_n = 1
u_matrix = QuantumCircuit(psi_n, name='U')
u_matrix.p(pi/4, 0)
qc = QuantumCircuit(n + psi_n, n)
qc.x(n)
qpe(qc, n, u_matrix, psi_n)
qc.draw(output='mpl')
plt.show()

qasm_sim = Aer.get_backend('qasm_simulator')
shots = 2048
t_qpe = transpile(qc, qasm_sim)
t_qpe.draw(output='mpl')
plt.show()
qobj = assemble(t_qpe, shots=shots)
results = qasm_sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)
plt.show()

