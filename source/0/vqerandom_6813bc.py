# https://github.com/harshdeeps036/FirstGame/blob/55c2d5c4ba6140fa450e98a93191603c7ab24d39/VQErandom.py

# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumRegister, ClassicalRegister, execute
from qiskit import QuantumCircuit, execute, Aer
from qiskit.tools.jupyter import *
import matplotlib.pyplot as plt
import numpy as np
from math import pi


def real_map(value, leftMin, leftMax, rightMin, rightMax):
    # Maps one range to another
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


# Variational Quantum Random Number generator
def VarQRandom(a, b, theta, depth, qubits):
    
    q = QuantumRegister(qubits, 'q')
    circ = QuantumCircuit(q)
    c0 = ClassicalRegister(2, 'c0')
    circ.add_register(c0)
    
    for j in range(depth):
        
        for i in range(qubits):
            circ.h(q[i])
            #rotating the axis by an angle theta along y-axis
            circ.ry(theta, q[i])
            #rotating the axis by an angle theta along x-axis
            circ.rx(theta, q[i])

    for i in range(qubits):
        circ.measure(q[i], c0)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    output = result.get_statevector(circ, decimals=5)

    n1 = 0
    n2 = 0
    n3 = 0
    for i in range( output.size ):
        if abs(output[i]) != 0:
            n1 = i
            n2 = np.real(output[i])
            n3 = np.imag(output[i])
    
    y = real_map(n1+n2+n3, -qubits, len(output)-1+qubits, a, b)
    return y