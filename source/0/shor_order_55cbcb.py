# https://github.com/leonardompp/pfcf_codes/blob/6698a4c1cbb9d8fa3c59a7781b081bfc716a516c/src/shor_order.py
import argparse
import textwrap

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile

from qiskit.tools.visualization import plot_histogram

from qiskit.circuit.library import QFT

from qiskit.extensions import UnitaryGate

import matplotlib.pyplot as plt
import numpy as np
import math

def generate_base_matrix(a, N, eigen_qubits):
    # Build the matrix, which must support 2**eigen_qubits elements
    u = np.zeros([2 ** eigen_qubits, 2 ** eigen_qubits], dtype=int)

    # Elements in 0 <= i <= N-1
    for i in range(N):
        u[a * i % N][i] = 1
    # For the rest, it is irrelevant. We'll not deal with these states
    for i in range(N, 2 ** eigen_qubits):
        u[i][i] = 1

    return u

def main(a, N):
    # Acknowledge user input
    print(f"Find order of element {a} in Z_{N}")

    # Calculate size of the eigenstate register
    eigen_qubits = math.ceil(np.log2(N))
    # Calculate size of the evaluation register
    eval_qubits = math.ceil(np.log2(pow(N, 2)))
    print(f"Eigenstate qubits: {eigen_qubits}")
    print(f"Evaluation qubits: {eval_qubits}")

    # Build circuit structure
    eigen_register = []
    for i in range(eigen_qubits):
        eigen_register.append(QuantumRegister(1, f"eigen{i}"))
    eval_register = []
    for i in range(eval_qubits):
        eval_register.append(QuantumRegister(1, f"eval{i}"))
    qc = QuantumCircuit(*eigen_register, *eval_register, ClassicalRegister(eval_qubits))

    # Circuit creation
    # Start state |1> on eigenstate register
    qc.x(0)

    # Start hadamards on evaluation register
    for i in range(eval_qubits):
        qc.h(i + eigen_qubits)

    # Build controlled rotations
    u = generate_base_matrix(a, N, eigen_qubits)
    for i in range(eval_qubits):
        power = pow(2, i)
        compound_u = np.linalg.matrix_power(u, power)
        compound_u_gate = UnitaryGate(compound_u)
        compound_u_gate.name = f"CU{2 ** i}"
        controlled_u_gate = compound_u_gate.control()
        qc.append(controlled_u_gate, [i + eigen_qubits, *list(range(eigen_qubits))])

    # Inverse qft
    qft = QFT(num_qubits=eval_qubits)
    qft_inverse = qft.inverse()
    qc.append(qft_inverse, [i + eigen_qubits for i in range(eval_qubits)])

    # Assign measurements
    for i in range(eval_qubits):
        qc.measure(i + eigen_qubits, i)

    # Draw
    qc.draw(output='mpl', initial_state=True, reverse_bits=True)  # Reverse for visualization, highest is MSQ

    # Compile and run
    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, simulator)

    result = simulator.run(transpiled).result()
    data = result.get_counts()
    # Pass data to decimal for easier inspection
    data_dec = dict()
    for bin_key in data.keys():
        data_dec[int(bin_key, 2)] = data[bin_key]
    print(data_dec)
    plot_histogram(data_dec, title=f"Shor results for N={N} a={a} - {eval_qubits} eval qubits")

    # Show all images
    plt.show()

if __name__ == "__main__":
    # Script instruction
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            Performs quantum core of Shor's order-finding algorithm on element a in group Z_N \n 
            Shows the probabilistic measurements of running QPE(U, |1>), which can then be used to estimate the order
            """),
        epilog=textwrap.dedent(
            """
            Example usages: 
            python src/shor_order.py -a 5 -N 13
            """),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-a', nargs=1, type=int, help='Element a in Z_N whose order we wish to find', required=True)
    parser.add_argument('-N', nargs=1, type=int, help='Integer N defining the modular algebra size', required=True)
    args = parser.parse_args()

    a = abs(args.a[0])
    N = abs(args.N[0])

    if N == 0 or math.gcd(a, N) != 1 or a > N:
        print("Invalid arguments")
    else:
        main(a, N)
