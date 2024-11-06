# https://github.com/francescozatelli/AQCM/blob/b595a60749cab43650c8b6c5c24ec31fa9dc4dcc/aqcm_circuits.py
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, assemble, transpile
from qiskit.circuit import Parameter


def universal_qcm(qc, input, ancilla1, ancilla2):
    """
    Basic circuit for universal cloning machine. The copies will be on the input qubit and on the first ancilla qubit.
    The qubits of the two copies are returned.
    No SWAP gate is used here (all the qubits must be connected).
    Ideal fidelity F=5/6=0.8333...
    """
    # Rotation angles used to prepare the ancillae
    theta1 = np.arccos(1 / np.sqrt(5))
    theta2 = np.arccos(np.sqrt(5) / 3)
    theta3 = np.arccos(2 / np.sqrt(5))

    # Preparation of the ancillae
    qc.ry(theta1, ancilla1)
    qc.cx(ancilla1, ancilla2)
    qc.ry(theta2, ancilla2)
    qc.cx(ancilla2, ancilla1)
    qc.ry(theta3, ancilla1)

    # Actual copying
    qc.cx(input, ancilla1)
    qc.cx(input, ancilla2)
    qc.cx(ancilla1, input)
    qc.cx(ancilla2, input)

    return input, ancilla1


def universal_qcm_SWAP(qc, input, ancilla1, ancilla2):
    """
    Basic circuit for universal cloning machine. The copies will be on the input qubit and on the first ancilla qubit.
    With this circuit only the first ancilla has to be connected to both the other qubits.
    A swap gate is then used in order to perform two qubits gates between the input state and both the first ancilla.
    The qubits of the two copies are returned.
    This circuit can be used even if not all the qubits are mutually connected.
    Ideal fidelity F=5/6=0.8333...
    """

    # Rotation angles used to prepare the ancillae
    theta1 = np.arccos(1 / np.sqrt(5))
    theta2 = np.arccos(np.sqrt(5) / 3)
    theta3 = np.arccos(2 / np.sqrt(5))

    # Preparation of the ancillae
    qc.ry(theta1, ancilla1)
    qc.cx(ancilla1, ancilla2)
    qc.ry(theta2, ancilla2)
    qc.cx(ancilla2, ancilla1)
    qc.ry(theta3, ancilla1)

    # Swap the input qubit and the ancilla1 qubit
    #qc.swap(input, ancilla1)
    qc.cx(input, ancilla1)
    qc.cx(ancilla1, input)
    qc.cx(input, ancilla1)

    # Actual copying
    qc.cx(ancilla1, input)
    qc.cx(ancilla1, ancilla2)
    qc.cx(input, ancilla1)
    qc.cx(ancilla2, ancilla1)

    return input, ancilla1


def phase_covariant_qcm(qc, input, ancilla1, ancilla2):
    """
    Phase covariant cloning machine. Optimally clones the states on the xz equator.
    The copies are made on the two ancilla qubits, which are returned. Only the input qubit needs to be connected to the other two.
    Ideal fidelity F=1/2+1/sqrt(8)=0.854...
    """
    # Rotation angle used to prepare the ancillae
    theta = np.pi / 4

    # Preparation of the ancillae
    qc.ry(theta, ancilla1)
    qc.ry(theta, ancilla2)

    # Actual copying
    qc.cx(input, ancilla2)
    qc.cx(input, ancilla1)
    qc.cx(ancilla2, input)
    qc.cx(ancilla1, input)

    return ancilla1, ancilla2


def economical_qcm(qc, input, ancilla1):
    """
    Economical Phase covariant cloning machine. Optimally clones the states on the xz equator.
    The copies are made on the two ancilla qubits, which are returned. Only the input qubit needs to be connected to the other two.
    Ideal fidelity F=(1+cos(pi/4))/2=0.8535
    """

    qc.sxdg(input)  # rotation about x of -pi/2
    qc.ch(input, ancilla1)
    qc.cx(ancilla1, input)
    qc.sx(input)
    qc.sx(ancilla1)

    return input, ancilla1