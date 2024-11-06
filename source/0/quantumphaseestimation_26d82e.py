# https://github.com/huwenqing0606/Quantum-Computing-using-Qiskit/blob/a198fb26a895b04d54f7f6065ef691ba919bd1ba/QuantumPhaseEstimation.py
from numpy import pi
# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
# import basic plot tools
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
# import inverse Quantum Fourier Transform
from QuantumFourierTransform import QFourier_inverse


# the circuit for Quantum Phase Estimation
def QuantumPhaseEstimation(qc, t, theta):
    # add H gates to qubit 0, 1, ..., t-1
    for qubit in range(t):
        qc.h(qubit)
    qc.barrier()
    # add powers of unitary gates to qubit t
    for i in range(t):
        number_power_unitary = t-1-i
        for j in range(2**number_power_unitary):
            control_qubit = i
            target_qubit = t
            qc.cp(2*pi*theta, control_qubit, target_qubit)
        qc.barrier()
    # inverse Fourier transform on the first t qubits
    QFourier_inverse(qc, t)
    # measure the first t qubits
    qc.measure(range(t), range(t))

    return qc


if __name__ == '__main__':
    t = 3
    theta = 1/3
    # initialization
    qc = QuantumCircuit(t+1, t)
    qc.x(t)
    qc.barrier()
    # build the circuit for quantum phase estimation
    QuantumPhaseEstimation(qc, t, theta)
    qc.draw(output='mpl')
    plt.show()

    aer_sim = Aer.get_backend('aer_simulator')
    shots = 2048
    qobj = assemble(qc, shots=shots)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()
    plot_histogram(counts)
    plt.show()