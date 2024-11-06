# https://github.com/vipulsnk/quantum_tic_tac_toe/blob/f854b3b58e65d5738a45a2ee1d6ec45567799651/core/quantum.py
import matplotlib as mpl
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import AerSimulator
import matplotlib.pyplot as plt
import os


def create_state():
    circuit = QuantumCircuit(1, 1)
    # circuit.h([0])
    return circuit


def draw_circuit(circuit):
    fig = circuit.draw(output="mpl")
    print(os.getcwd())
    # plt.savefig("./core/snaps/circuit.jpeg")
    plt.savefig("./static/img/circuit.jpeg")
    plt.close(fig)
    # plt.close('all')


def find_bit_index_by_name(circuit, qr_name):
    for i, bit in enumerate(circuit.qubits):
        if bit.register.name == qr_name:
            return i
    return -1


def add_measure(circuit, qr_name):
    ind = find_bit_index_by_name(circuit, qr_name)
    circuit.measure(ind, ind)


def create_superposed_position(circuit, row1, col1, row2, col2):
    if (2 * row1 + col1) > (2 * row2 + col2):
        row1 = row1 ^ row2
        row2 = row1 ^ row2
        row1 = row1 ^ row2
        col1 = col1 ^ col2
        col2 = col1 ^ col2
        col1 = col1 ^ col2
    qr_name = "q" + str(row1) + str(col1) + "_" + str(row2) + str(col2)
    if find_bit_index_by_name(circuit, qr_name) == -1:
        qr = QuantumRegister(1, qr_name)
        circuit.add_register(qr)
        cr_name = "c" + str(row1) + str(col1) + "_" + str(row2) + str(col2)
        cr = ClassicalRegister(1, cr_name)
        circuit.add_register(cr)
        circuit.h([find_bit_index_by_name(circuit, qr_name)])
    return qr_name


# def remove_register(circuit):


def simulate(circuit):
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result()
    count = result.get_counts(circuit)
    return count

