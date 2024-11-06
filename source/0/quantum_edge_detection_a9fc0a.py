# https://github.com/Djmcflush/Quantum-Hackathon/blob/2207f91b58fb7a74102e3303c8c8f79fb1291bac/quantum_edge_detection.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute
#from qiskit.aqua.circuits.fourier_transform_circuits import FourierTransformCircuits
from math import pi
from qiskit import Aer

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cu1(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit
def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    # First we create a QFT circuit of the correct size:
    qft_circ = qft(QuantumCircuit(n), n)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the first n qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows us to see the individual gates
    
def FourierTransformCircuits(circuit, qubits, inverse=False):
    if inverse:
        inverse_qft(circuit, qubits)
    else:
        qft(circuit,qubits)


def quantum_adder(circuit, epsilon):
    qubits = circuit.qubits
    n_qubits = circuit.num_qubits
    FourierTransformCircuits(circuit, n_qubits)
    for i in range(n_qubits):
        circuit.u1(float(2 * pi * epsilon)/2**(i + 1), qubits[n_qubits - i - 1])
    FourierTransformCircuits(circuit, n_qubits, inverse=True)


def quantum_edge_detection(circuit):        #gives you the quantum state where you have to measure the ancilla and obtain 0
    qubits = circuit.qubits
    ancilla = qubits[0]
    circuit.h(ancilla)
    quantum_adder(circuit, -1)
    circuit.h(ancilla)
    circuit.x(ancilla)

#/!\ Vertical and horizontal encoding
'''
circuit.measure(ancilla, clbit)

backend = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, backend)
sim_result = job_sim.result()

print(sim_result.get_counts(circuit))
'''
