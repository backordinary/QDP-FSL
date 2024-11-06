# https://github.com/leonardompp/pfcf_codes/blob/6698a4c1cbb9d8fa3c59a7781b081bfc716a516c/src/qpe.py
import argparse
import textwrap

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile

from qiskit.tools.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.extensions import Initialize, UnitaryGate
from qiskit.circuit.library import QFT

import numpy as np
import matplotlib.pyplot as plt

import math


def main(eval_qubits):
    #########################################################################
    ################ CHANGE HERE FOR DIFFERENT U AND |phi> ##################
    #########################################################################
    # Examples
    # U = np.array([[0, 1], [-1, 0]]) phi = (1/np.sqrt(2))*np.array([1, 1j])
    # U = np.array([[1, 0], [0, np.exp(2j*np.pi/3)]]) phi = np.array([0, 1])
    U = np.array([[0, 1], [-1, 0]])
    phi = (1/np.sqrt(2))*np.array([1, 1j])
    #########################################################################

    # Acknowledge user input
    print(f"Operator U:")
    print(f"{U}")
    print(f"Eigenstate |phi>:")
    print(f"{phi}")

    # Qubits in eigenstate register
    eigen_qubits = int(np.log2(phi.size))
    print(f"Number of eigenstate qubits needed   : {eigen_qubits}")
    print(f"Number of evaluation qubits requested: {eval_qubits}")

    # Build circuit structure
    eigen_register = []
    for i in range(eigen_qubits):
        eigen_register.append(QuantumRegister(1, f"eigen{i}"))
    eval_register = []
    for i in range(eval_qubits):
        eval_register.append(QuantumRegister(1, f"eval{i}"))
    qc = QuantumCircuit(*eigen_register, *eval_register, ClassicalRegister(eval_qubits))

    # Initialize eigenstate
    init_gate = Initialize(Statevector(phi))
    init_gate.name = "$|\phi>$"
    qc.append(init_gate, list(range(eigen_qubits)))

    # Initialize hadamards on eval qubits
    for i in range(eval_qubits):
        qc.h(i + eigen_qubits)

    # Add rotation gates
    for i in range(eval_qubits):
        power = pow(2, i)
        compound_u = np.linalg.matrix_power(U, power)
        compound_u_gate = UnitaryGate(compound_u)
        compound_u_gate.name = f"$CU^{2**i}$"
        controlled_u_gate = compound_u_gate.control()
        qc.append(controlled_u_gate, [i+eigen_qubits, *list(range(eigen_qubits))])

    # Inverse qft
    qft = QFT(num_qubits=eval_qubits)
    qft_inverse = qft.inverse()
    qc.append(qft_inverse, [i+eigen_qubits for i in range(eval_qubits)])

    # Assign measurements
    for i in range(eval_qubits):
        qc.measure(i+eigen_qubits, i)

    # Draw circuit
    qc.draw(output='mpl', initial_state=True, reverse_bits=True)  # Reverse for visualization, highest is MSQ

    # Compile and run
    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, simulator)

    result = simulator.run(transpiled).result()
    data = result.get_counts()
    plot_histogram(data, title=f"QPE $(U, |\phi>)$ on {eval_qubits} eval qubits")

    # Show all images
    plt.show()

if __name__ == "__main__":
    # Script instruction
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            This script performs quantum phase estimation for some unitary operator U and eigenstate |phi> \n
            Number of qubits in eigenstate register is calculated automatically \n
            IMPORTANT: THE VALUES OF U AND |phi> HAVE TO BE CHANGED INSIDE THE SCRIPT
            """),
        epilog=textwrap.dedent(
            """
            Example usages: 
            python src/qpe.py -n 2
            """),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', nargs=1, type=int, help='Number of qubits in evaluation register', required=True)
    args = parser.parse_args()

    n = abs(args.n[0])
    main(n)

