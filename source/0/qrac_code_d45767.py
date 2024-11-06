# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/SuperDenseCoding%20-%20RAC/qrac_code.py
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/nathanbunch/qiskit-sdk-py'))

# useful additional packages
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from math import pi

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

backend = 'local_qasm_simulator' # the device to run on
shots = 1024    # the number of shots in the experiment 

#to record the rotation number for encoding 00, 10, 11, 01
rotationNumbers = {"00":1, "10":3, "11":5, "01":7}

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# Creating registers
# qubit for encoding 2 bits of information
qr = Q_program.create_quantum_register("qr", 1)
# bit for recording the measurement of the qubit
cr = Q_program.create_classical_register("cr", 1)

# dictionary for encoding circuits
encodingCircuits = {}
# Quantum circuits for encoding 00, 10, 11, 01
for bits in ("00", "01", "10", "11"):
    circuitName = "Encode"+bits
    encodingCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
    encodingCircuits[circuitName].u3(rotationNumbers[bits]*pi/4.0, 0, 0, qr[0])
    encodingCircuits[circuitName].barrier()

# dictionary for decoding circuits
decodingCircuits = {}
# Quantum circuits for decoding the first and second bit
for pos in ("First", "Second"):
    circuitName = "Decode"+pos
    decodingCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
    if pos == "Second": #if pos == "First" we can directly measure
        decodingCircuits[circuitName].h(qr[0])
    decodingCircuits[circuitName].measure(qr[0], cr[0])

#combine encoding and decoding of QRACs to get a list of complete circuits
circuitNames = []
for k1 in encodingCircuits.keys():
    for k2 in decodingCircuits.keys():
        circuitNames.append(k1+k2)
        Q_program.add_circuit(k1+k2, encodingCircuits[k1]+decodingCircuits[k2])

print("List of circuit names:", circuitNames) #list of circuit names
Q_program.get_qasms(circuitNames) #list qasms codes

results = Q_program.execute(circuitNames, backend=backend, shots=shots)
print("Experimental Result of Encode01DecodeFirst")
plot_histogram(results.get_counts("Encode01DecodeFirst"))  #We should measure "0" with probability 0.85
print("Experimetnal Result of Encode01DecodeSecond")
plot_histogram(results.get_counts("Encode01DecodeSecond")) #We should measure "1" with probability 0.85
print("Experimental Result of Encode11DecodeFirst")
plot_histogram(results.get_counts("Encode11DecodeFirst"))  #We should measure "1" with probability 0.85
print("Experimental Result of Encode11DecodeSecond")
plot_histogram(results.get_counts("Encode11DecodeSecond")) #We should measure "1" with probability 0.85
