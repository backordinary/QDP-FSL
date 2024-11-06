# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutoriale-QISKit/01-getting_started/visualization.py
import sys
import os

# import qiskit from qiskit-sdk-py folder
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'qiskit-sdk-py'))
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
except:
    qx_config = {
        "APItoken":"da2a4002660558a35103a600bcbda7fe438cea629a6be98969ea5e367c091b6815e624bd86b6207121bd97fef79c22033318a4402eeafcbd04b021fd80f5a195",
        "url":"https://quantumexperience.ng.bluemix.net/api"
    }

import matplotlib.pyplot as plt 
import numpy as np 
from pprint import pprint
from scipy import linalg as la 

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, QISKitError
from qiskit import available_backends, execute, register, get_backend

from qiskit.tools.visualization import plot_histogram, plot_state

def ghz_state(q, c, n):
    # Create a GHZ state
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    for i in range(n-1):
        qc.cx(q[i], q[i+1])
    return qc

def superposition_state(q, c):  
    # Create a superposition state
    qc = QuantumCircuit(q, c)
    qc.h(q)
    return qc

# Build the quantum circuit. We are going to build two circuits a GHZ over 3 qubits and a superposition over all 3 qubits

n = 3 # number of qubits
q = QuantumRegister(n)
c = ClassicalRegister(n)

# quantum circuit to make a GHZ state
ghz = ghz_state(q, c, n)

# quantum circuit to make a superposition state
superposition = superposition_state(q, c)

measure_circuit = QuantumCircuit(q, c)
measure_circuit.measure(q, c)

# execute the quantum circuits
backend = 'local_qasm_simulator' 
circuits = [ghz+measure_circuit, superposition+measure_circuit]
shots = 4096
job = execute(circuits, backend=backend, shots=shots)

plot_histogram(job.result().get_counts(circuits[0]))
plot_histogram(job.result().get_counts(circuits[1]),15)