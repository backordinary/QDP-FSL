# https://github.com/Ming2027/RussianPeasantMultiplication/blob/0d724b98bce159c6c0c4e390230d949610604f1c/bernstein_vazirani.py
# initialization
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute


def quantum_calculate(s):
    nQubits = math.floor(math.log2(s)+1) # number of physical qubits used to represent s
    s = s % 2**(nQubits)
    # Creating registers
    # qubits for querying the oracle and finding the hidden integer
    qr = QuantumRegister(nQubits)
    # bits for recording the measurement on qr
    cr = ClassicalRegister(nQubits)

    bvCircuit = QuantumCircuit(qr, cr)
    barriers = True

    # Apply Hadamard gates before querying the oracle
    for i in range(nQubits):
        bvCircuit.h(qr[i])

    # Apply barrier
    if barriers:
        bvCircuit.barrier()

    # Apply the inner-product oracle
    for i in range(nQubits):
        if (s & (1 << i)):
            bvCircuit.z(qr[i])
        else:
            bvCircuit.iden(qr[i])

    # Apply barrier
    if barriers:
        bvCircuit.barrier()

    #Apply Hadamard gates after querying the oracle
    for i in range(nQubits):
        bvCircuit.h(qr[i])

    # Apply barrier
    if barriers:
        bvCircuit.barrier()

    # Measurement
    bvCircuit.measure(qr, cr)

    # use local simulator
    backend = BasicAer.get_backend('qasm_simulator')
    shots = 1
    results = execute(bvCircuit, backend=backend, shots=shots).result()
    answer = results.get_counts()
    amw = (*answer,)
    return amw[0]