# https://github.com/ccicconetti/qiskit-tests/blob/8e362a1d041b071b64b9630201e0bb56246c735a/tutorial/superdense-coding.py
"""Superdense example inspired from the Qiskit Textbook"""

import operator
import math
import random
import matplotlib.pyplot as plt

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    BasicAer
)

from utils import (
    NoiseModelWrapper,
    decode_message
)

#
# Configuration
#

possible_messages = ['00', '01', '10', '11']
experiment_type = 'simulator-noise'
shots = 1024
msg_to_send = random.choice(possible_messages)

#
# Execution
#

# Load noisy gates, if required
noise_wrapper = None
if experiment_type == 'simulator-noise':
    noise_wrapper = NoiseModelWrapper('ibmq_essex')

# Initialize circuit and registers
qr = QuantumRegister(2)    # Protocol needs Bell pair of qubits
cr = ClassicalRegister(2)  # Classical bits for the message
qc = QuantumCircuit(qr, cr)

# Create Bell pair
qc.h(0)
qc.cx(0, 1)

# Encode message [done by Alice using qubit 0 only]
print("message to send: {}".format(msg_to_send))
if msg_to_send == '00':
    pass
elif msg_to_send == '01':
    qc.z(0)
elif msg_to_send == '10':
    qc.x(0)
elif msg_to_send == '11':
    qc.z(0)
    qc.x(0)
else:
    raise Exception("Invalid message to be sent: {}".format(msg_to_send))
qc.barrier()

# Decode message [done by Bob using qbit 1 only]
qc.cx(0, 1)
qc.h(0)
qc.barrier()
qc.measure(qr, cr)

# Print circuit
print(qc.draw(output='text'))

#
# Test circuit
#
if experiment_type == 'simulator-statevector':
    # Run statevector simulation
    backend = BasicAer.get_backend('statevector_simulator')
    out_vector = execute(qc, backend).result().get_statevector(decimals=2)
    assert len(out_vector) == 4
    for i in range(len(possible_messages)):
        if abs(out_vector[i]) > 0.5:
            print("received message {}".format(possible_messages[i]))
            break

else:
    if experiment_type == 'simulator-qasm':
        backend = BasicAer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=shots).result()

    elif experiment_type == 'simulator-noise':
        assert noise_wrapper is not None
        result = noise_wrapper.execute(qc)
    
    # Find measurement with maximum probability
    print(result.get_counts())
    decode_message(result, print_message=True)
