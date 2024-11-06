# https://github.com/lionelchg/LesAtomesDeSavoie/blob/9e0c37ce2ea634446b0406ceea90d9432461ae86/qiskit/grover/two_bits.py
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
    n = 2
    grover_circuit = QuantumCircuit(n)

    # Initialize the state
    grover_circuit = initialize_s(grover_circuit, [0, 1])

    # Oracle with controlled Z (specific to 2 bits)
    grover_circuit.cz(0, 1)

    # Diffusion operator (U_s)
    grover_circuit.h([0, 1])
    grover_circuit.z([0, 1])
    grover_circuit.cz(0, 1)
    grover_circuit.h([0, 1])
    grover_circuit.measure_all()
    grover_circuit.draw(output='mpl', filename='figures/2bits_circuit')

    # Statevector
    sv_sim = Aer.get_backend('statevector_simulator')
    qobj = assemble(grover_circuit)
    result = sv_sim.run(qobj).result()
    statevec = result.get_statevector()

    # Running the simulation
    run_sim(grover_circuit, 'figures/hist_2bits_sim')

    # Run on IBMQ
    run_ibmq(grover_circuit, 'figures/hist_2bits_ibmq')