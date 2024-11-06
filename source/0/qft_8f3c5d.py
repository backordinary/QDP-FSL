# https://github.com/Danny-sc/QFT_multiplier/blob/057b8450b8e21e67eeb6a2e444eb6fe1e950b052/qft.py
from qiskit import(QuantumCircuit)
from numpy import pi


def qft(circuit, n):
    """
    Given a quantum circuit and a positive integer n, this function performs 
    the Quantum Fourier Transform on the first n qubits of the circuit.

    Parameters
    ----------
    circuit : QuantumCircuit object
        The circuit on which to perform the Quantum Fourier Transform.
    n : int
        The number of qubits on which to perform the Quantum Fourier Transform.

    Returns
    -------
    circuit : QuantumCircuit object
        The original circuit with the Quantum Fourier Transform applied to the 
        first n qubits.
    """
    # The circuit should have at least one qubit
    if n == 0: 
        return circuit
    # n is reduced by one, since indexes start from zero in Python
    n -= 1 
    # Apply the Hadamard gate to the most significant qubit
    circuit.h(n) 
    # Apply the appropriate conditional rotation for each less significant qubit.
    # To rotate the second most significant qubit, we call qft_rotations again 
    # (note that n was already reduced by one)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)     
    qft(circuit, n)
    return circuit
    
def inverse_qft(circuit, n):
    """
    Given a quantum circuit and a positive integer n, this function performs 
    the inverse of the Quantum Fourier Transform on the first n qubits of the 
    circuit.

    Parameters
    ----------
    circuit : QuantumCircuit object
        The circuit on which to perform the inverse of the Quantum Fourier 
        Transform.
    n : int
        The number of qubits on which to perform the inverse of the 
        Quantum Fourier Transform.

    Returns
    -------
    circuit : QuantumCircuit object
        The original circuit with the inverse of the Quantum Fourier Transform 
        applied to the first n qubits.
    """
    # Build a Quantum Fourier Transform circuit of the correct size with `qft`
    qft_circ = qft(QuantumCircuit(n), n)     
    # Take the inverse of the built circuit
    invqft_circ = qft_circ.inverse() 
    # Append the inverse to the first n qubits in the original circuit
    circuit.append(invqft_circ, circuit.qubits[:n]) 
    return circuit

