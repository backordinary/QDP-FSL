# https://github.com/ioanasv/quantumInspire/blob/9b8b029a79032c660d62d61700f77f9f28e7ceaa/quantuminspire/docs/example_qiskit_entangle.py
"""Example usage of the Quantum Inspire backend with the QisKit SDK.

A simple example that demonstrates how to use the SDK to create
a circuit to create a Bell state, and simulate the circuit on
Quantum Inspire.

For documentation on how to use QisKit we refer to
[https://qiskit.org/](https://qiskit.org/).

Specific to Quantum Inspire is the creation of the QI instance, which is used to set the authentication of the user and
provides a Quantum Inspire backend that is used to execute the circuit.


Copyright 2018-19 QuTech Delft. Licensed under the Apache License, Version 2.0.
"""
import os
from getpass import getpass
from math import sqrt

from quantuminspire.src.quantuminspire.credentials import load_account, get_token_authentication, \
    get_basic_authentication

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute

from quantuminspire.src.quantuminspire.qiskit import QI

QI_EMAIL = os.getenv('QI_EMAIL')
QI_PASSWORD = os.getenv('QI_PASSWORD')
QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')


def get_authentication():
    """ Gets the authentication for connecting to the Quantum Inspire API."""
    token = load_account()
    if token is not None:
        return get_token_authentication(token)
    else:
        if QI_EMAIL is None or QI_PASSWORD is None:
            print('Enter email:')
            email = input()
            print('Enter password')
            password = getpass()
        else:
            email, password = QI_EMAIL, QI_PASSWORD
        return get_basic_authentication(email, password)


def entangler(initial_state, circuit):

    circuit.initialize(initial_state, 0)
    zero_state = [1, 0]
    circuit.initialize(zero_state, 1)
    circuit.initialize(zero_state, 2)
    circuit.initialize(zero_state, 3)
    circuit.initialize(zero_state, 4)

    circuit.h(q[1])
    circuit.cx(q[1], q[2])
    circuit.cx(q[2], q[3])
    circuit.cx(q[3], q[4])
    circuit.cx(q[0], q[1])

    circuit.measure(1, b1)

    circuit.x(q[1]).c_if(b1, 1)
    circuit.x(q[2]).c_if(b1, 1)
    circuit.x(q[3]).c_if(b1, 1)
    circuit.x(q[4]).c_if(b1, 1)
    return circuit


def disentangler(circuit):
    circuit.h(q[2])
    circuit.h(q[3])
    circuit.h(q[4])

    circuit.measure(q[2], b2)
    circuit.measure(q[3], b3)
    circuit.measure(q[4], b4)

    circuit.z(q[0]).c_if(b2, 1)
    circuit.z(q[0]).c_if(b3, 1)
    circuit.z(q[0]).c_if(b4, 1)

    circuit.x(q[2]).c_if(b2, 1)
    circuit.x(q[3]).c_if(b3, 1)
    circuit.x(q[4]).c_if(b4, 1)
    return circuit

def teleport(state):
    q = QuantumRegister(4)
    b0 = ClassicalRegister(1)
    b1 = ClassicalRegister(1)
    b2 = ClassicalRegister(1)
    b3 = ClassicalRegister(1)
    qc = QuantumCircuit(q, b0, b1, b2, b3)
    qc.initialize(state, 0)
    zero_state = [1, 0]
    qc.initialize(zero_state, 1)
    qc.initialize(zero_state, 2)
    qc.initialize(zero_state, 3)

    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[0], q[1])
    qc.measure(q[1], b1)
    qc.x(q[1]).c_if(b1, 1)
    qc.x(q[2]).c_if(b1, 1)
    qc.h(q[0])
    qc.measure(q[0], b0)
    qc.x(q[0]).c_if(b0, 1)
    qc.x(q[2]).c_if(b0, 1)
    qc.swap(q[2], q[3])
    return qc


def nonlocal_cnot(control, target):
    q = QuantumRegister(4)
    b0 = ClassicalRegister(1)
    b1 = ClassicalRegister(1)
    b2 = ClassicalRegister(1)
    b3 = ClassicalRegister(1)
    qc = QuantumCircuit(q, b0, b1, b2, b3)
    qc.initialize(control, 0)
    qc.initialize(target, 3)
    zero_state = [1, 0]
    qc.initialize(zero_state, 1)
    qc.initialize(zero_state, 2)

    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[0], q[1])
    qc.measure(q[1], b1)
    qc.x(q[1]).c_if(b1, 1)
    qc.x(q[2]).c_if(b1, 1)
    qc.cx(q[2], q[3])
    qc.h(q[2])
    qc.measure(q[2], b2)
    qc.z(q[0]).c_if(b2, 1)
    qc.x(q[2]).c_if(b2, 1)

    return qc

if __name__ == '__main__':
    authentication = get_authentication()
    QI.set_authentication(authentication, QI_URL)
    qi_backend = QI.get_backend('QX single-node simulator')

    # TRY ENTANGLER AND DISENTANGLER
    # q = QuantumRegister(5)
    #
    # b0 = ClassicalRegister(1)
    # b1 = ClassicalRegister(1)
    # b2 = ClassicalRegister(1)
    # b3 = ClassicalRegister(1)
    # b4 = ClassicalRegister(1)
    #
    # circuit = QuantumCircuit(q, b0, b1, b2, b3, b4)
    #
    # initial_state = [1/sqrt(3), sqrt(2)/sqrt(3)]
    # entangled_circuit = entangler(initial_state, circuit)
    # disentangled = disentangler(entangled_circuit)
    # q2 = disentangled.qubits
    # b = disentangled.clbits
    # circuit.measure(q2[0], b[0])
    # circuit.measure(q2[1], b[1])
    # circuit.measure(q2[2], b[2])
    # circuit.measure(q2[3], b[3])
    # circuit.measure(q2[4], b[4])

    # TRY TELEPORT
    # state = [1/sqrt(2), 1/sqrt(2)]
    # circuit = teleport(state)
    # q = circuit.qubits
    # b = circuit.clbits
    # circuit.measure(q[0], b[0])
    # circuit.measure(q[1], b[1])
    # circuit.measure(q[2], b[2])
    # circuit.measure(q[3], b[3])

    # TRY CNOT
    # control = [1/sqrt(2), 1/sqrt(2)]
    # target = [1, 0]
    # circuit = nonlocal_cnot(control, target)
    # q = circuit.qubits
    # b = circuit.clbits
    # circuit.measure(q[0], b[0])
    # circuit.measure(q[1], b[1])
    # circuit.measure(q[2], b[2])
    # circuit.measure(q[3], b[3])


    qi_job = execute(circuit, backend=qi_backend, shots=256)
    qi_result = qi_job.result()
    histogram = qi_result.get_counts(circuit)
    print('\nState\tCounts')
    [print('{0}\t\t{1}'.format(state, counts)) for state, counts in histogram.items()]
    # Print the full state probabilities histogram
    probabilities_histogram = qi_result.get_probabilities(circuit)
    print('\nState\tProbabilities')
    [print('{0}\t\t{1}'.format(state, val)) for state, val in probabilities_histogram.items()]
