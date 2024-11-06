# https://github.com/leonardompp/pfcf_codes/blob/6698a4c1cbb9d8fa3c59a7781b081bfc716a516c/src/qft.py
import argparse
import textwrap

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.tools.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector

from qiskit.circuit.library import QFT

import matplotlib.pyplot as plt

import math


def main(state, qubits):
    # Build the circuit
    q_register = []
    for i in range(qubits):
        q_register.append(QuantumRegister(1, f'q{i}'))
    qc = QuantumCircuit(*q_register)

    # Pass the state to binary and adapt registers
    state_bin = bin(state)

    i = -1
    while state_bin[i] != 'b':
        if state_bin[i] == '1':
            qc.x(abs(i) - 1)
        i -= 1

    # Show the state before qft in Bloch sphere
    plot_state = Statevector(qc)
    plot_bloch_multivector(plot_state, title=f"State |{state}> on {qubits} qubits", reverse_bits=True)

    # Add qft
    qfc = QFT(num_qubits=qubits, name='QFT')
    qc.append(qfc, list(range(qubits)))

    # Show the state after qft in Bloch sphere
    plot_state = Statevector(qc)
    plot_bloch_multivector(plot_state, title=f"State QFT|{state}> on {qubits} qubits", reverse_bits=True)

    # Draw circuit
    qc.draw(output='mpl', initial_state=True, reverse_bits=True)  # Reverse for visualization, highest is MSQ
    plt.show()

if __name__ == "__main__":
    # Script instruction
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            This script performs quantum Fourier transform QFT|j> for some state |j> in the computational basis of the system \n
            OBS: Superpositions are not yet implemented \n 
            """),
        epilog=textwrap.dedent(
            """
            Example usages: 
            python src/qft.py 7 -> yields QFT|7> on 3 qubits
            python src/qft.py 7 -n 4 -> yields QFT|7> on 4 qubits
            """),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', nargs=1, type=int, help='Number of qubits. Defaults to minimum needed')
    parser.add_argument('j', nargs=1, type=int, help='Element |j> of the computational basis')
    args = parser.parse_args()

    n = args.n
    j = abs(args.j[0]) # Take module for safety

    # No specific number of qubits was requested
    if n is None:
        if j != 0:
            n = math.ceil(math.log(j, 2))
            # For the times when j is a power of 2
            if pow(2, n) == j:
                n = n + 1
        else:
            n = 1
    else:
        n = n[0]

    # Check for the case when user passes a number of qubits
    if pow(2, n) > j:
        main(j, n)
    else:
        print(f"Invalid number of qubits {n} for state desired {j}")
