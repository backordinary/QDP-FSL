# https://github.com/leonardompp/pfcf_codes/blob/6698a4c1cbb9d8fa3c59a7781b081bfc716a516c/src/bell_state.py
import argparse
import textwrap

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.tools.visualization import plot_state_qsphere
from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt

def main(qubit1, qubit0):
    # Simple bell state creation
    qc = QuantumCircuit(QuantumRegister(1, "q0"), QuantumRegister(1, "q1"))

    # Circuit
    if qubit1 == 1:
        qc.x(1)
    if qubit0 == 1:
        qc.x(0)

    qc.h(1)
    qc.cx(1, 0)

    # Plot in mpl
    qc.draw(output='mpl', initial_state=True, reverse_bits=True)  # Reverse for visualization, highest is MSQ

    # Plot QSphere (not described in paper, but can be found in qiskit. Generally too complicated, but useful here)
    print(f"Psi_[{qubit1}{qubit0}] = (1/sqrt(2))*(|0{qubit0}> {'+' if qubit1 == 0 else '-'} |1{1^qubit0}>)")

    state = Statevector(qc)
    plot_state_qsphere(state)
    plt.show()





if __name__ == "__main__":
    # Script instruction
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
        """
        This script generates Bell states Psi_[ij] as shown in the thesis \n
        OBS: Superpositions are not yet implemented \n 
        """),
        epilog=textwrap.dedent(
        """
        Example usage: python src/bell_state.py -q1 1 -q0 0 -> Creates Psi_[10]
        """),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-q1', nargs=1, default=0, type=int, choices=[0, 1], help='State of qubit q1')
    parser.add_argument('-q0', nargs=1, default=0, type=int, choices=[0, 1], help='State of qubit q0')
    args = parser.parse_args()

    # Unpack the arguments
    q1 = args.q1[0]
    q0 = args.q0[0]

    main(q1, q0)

