# https://github.com/lionelchg/LesAtomesDeSavoie/blob/9e0c37ce2ea634446b0406ceea90d9432461ae86/qiskit/grover/three_bits.py
#initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from utils import create_dir
from run import run_sim, run_ibmq
from circuit import initialize_s, diffuser

if __name__ == '__main__':
    create_dir('figures/')
    qc = QuantumCircuit(3)
    qc.cz(0, 2)
    # The 2 qubit seems to be the leftmost one
    qc.cz(1, 2)
    oracle_ex3 = qc.to_gate()
    oracle_ex3.name = "U$_\omega$"

    # Create circuit
    n = 3
    grover_circuit = QuantumCircuit(n)
    grover_circuit = initialize_s(grover_circuit, [0, 1, 2])
    grover_circuit.append(oracle_ex3, [0, 1, 2])
    grover_circuit.append(diffuser(n), [0, 1, 2])
    grover_circuit.measure_all()
    grover_circuit.draw(output='mpl', filename='figures/3bits_circuit')

    # Run on simulator
    run_sim(grover_circuit, 'figures/hist_3bits_sim')

    # # Run on IBMQ
    run_ibmq(grover_circuit, 'figures/hist_3bits_ibmq')
