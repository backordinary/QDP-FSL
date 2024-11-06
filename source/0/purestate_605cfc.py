# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutorial_programming_IBM/localSimulator/1/pureState.py
import sys
import os
# Checking the version of PYTHO; we only support > 3.5
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

# import qiskit from qiskit-sdk-py folder 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'qiskit-sdk-py'))

# useful additional packages
import matplotlib.pyplot as plt 
#matplotlib inline
import numpy as np 
from scipy import linalg as la 

# importing the QISKit
from qiskit import QuantumProgram, QuantumCircuit
import Qconfig

# import state tomography functions
from qiskit.tools.visualization import plot_histogram, plot_state

# Build the quantum circuit.
# GHZ over 3 qubits
# superposition over all 3 qubits
Q_program = QuantumProgram()
n = 3 # number of qubits
q = Q_program.create_quantum_register('q', n)
c = Q_program.create_classical_register('c', n)

# quantum circuit to make a GHZ state
ghz = Q_program.create_circuit('ghz', [q], [c])
ghz.h(q[0])
ghz.cx(q[0], q[1])
ghz.cx(q[0], q[2])
ghz.s(q[0])
ghz.measure(q[0], c[0])
ghz.measure(q[1], c[1])
ghz.measure(q[2], c[2])

# quantum circuit to make a superposition state
superposition = Q_program.create_circuit('superposition', [q], [c])
superposition.h(q)
superposition.s(q[0])
superposition.measure(q[0], c[0])
superposition.measure(q[1], c[1])
superposition.measure(q[2], c[2])

circuits = ['ghz', 'superposition']

# execute the quantum circuit
backend = 'local_qasm_simulator' # the device to run on
result = Q_program.execute(circuits, backend=backend, shots=1000)

ground = np.zeros(2**n)
ground[0] = 1.0

state_superposition = np.dot(result.get_data('superposition')['unitary'], ground)
rho_superposition = np.outer(state_superposition, state_superposition.conj())

state_ghz = np.dot(result.get_data('ghz')['unitary'], ground)
rho_ghz = np.outer(state_ghz, state_ghz.conj())

plot_state(rho_ghz, 'city')