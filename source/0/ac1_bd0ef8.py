# https://github.com/qis-unipr/entanglement-verification/blob/1314022cb1441370bec39c7b6dc7018e099f4397/qiskit-test/AC1.py

import sys
from sys import argv
import os
import logging
import random

# Import the QISKit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QiskitError
from qiskit import IBMQ, execute, BasicAer
from qiskit.providers.ibmq import least_busy

# Authenticate for access to remote backends
try:
    IBMQ.load_account()
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a file with your personal token?
             For now, there's only access to local simulator backends...""")

try:
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)

    AL1circuit = QuantumCircuit(qr, cr)

    meas = random.getrandbits(1)
    if (meas == 1):
        # Beta00 measured in diag. basis, giving |++>
        print('Bell state measured in the diagonal basis')
        AL1circuit.h(qr[1])
        AL1circuit.h(qr[2])
    else:
        # Beta00 measured in comp. basis, leaving |00>
        print('Bell state measured in the computational basis')

    s = random.getrandbits(1)
    print('s = ' + str(s))
    if (s == 1):
        AL1circuit.h(qr[0])

    AL1circuit.cx(qr[0],qr[1])
    AL1circuit.h(qr[0])

    AL1circuit.cx(qr[1],qr[2])
    AL1circuit.cz(qr[0],qr[2])
    if (s == 1):
        AL1circuit.h(qr[2])

    AL1circuit.measure(qr[0], cr[0])
    AL1circuit.measure(qr[1], cr[1])
    AL1circuit.measure(qr[2], cr[2])

    image = AL1circuit.draw()
    print(image)

    if (sys.argv[1] == 'sim'):
        backend = BasicAer.get_backend('qasm_simulator')
    elif (sys.argv[1] == 'real'):
        backend = least_busy(IBMQ.get_provider(hub='ibm-q').backends(filters=lambda x: x.configuration().n_qubits == 5 and not x.configuration().simulator))
        print("Running on current least busy device: ", backend)
    job_sim = execute(AL1circuit, backend, shots=8000)
    result_sim = job_sim.result()
    print(result_sim.get_counts(AL1circuit))


except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
