# https://github.com/easyesplat/QiskitFall2022/blob/9d9776cfd0b2d47000fcad937611c61c7ba86f49/quantum_random.py
from qiskit import *
from qiskit.compiler import transpile, assemble
import numpy as np

def QRandom(a, b, qubits=2):
    # Create qubits
    q = QuantumRegister(qubits, 'q')
    # Create quantum circuit
    circ = QuantumCircuit(q)
    # Create classical registers to store measured qubits
    c0 = ClassicalRegister(2, 'c0')
    circ.add_register(c0)

    # Apply hadamard gate to each qubit
    for i in range(qubits):
        circ.h(q[i])

    # Measure all qubits
    for i in range(qubits):
        circ.measure(q[i], c0)

    # Establish qiskit backend
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    # Get vector that holds information on one qubit from a quantum system
    output = result.get_statevector(circ, decimals=5)

    # Assigns randomness values
    n1 = 0
    n2 = 0
    n3 = 0
    for i in range( output.dim ):
        if abs(output[i]) != 0:
            n1 = i
            n2 = np.real(output[i])
            n3 = np.imag(output[i])
    y = prac_map(n1+n2+n3, -qubits, len(output)-1+qubits, a, b) 
    return y

def prac_map(value, leftMin, leftMax, rightMin, rightMax):
    # Calculate width of left and right values
    l_span = leftMax - leftMin
    r_span = rightMax - rightMin

    # Convert the left value into a float
    valueScaled = float(value - leftMin) / float(l_span)

    # Convert the float value into the right scale
    return rightMin + (valueScaled * r_span)