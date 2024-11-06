# https://github.com/brower/QuLAT/blob/30e53a6220d5b9ad08779ae755c1efce5167fcda/Hiroki/quantum_info/mitigation_gate.py
'''
Created on Mar 27, 2020

@author: kwibu
'''
import os, itertools, time
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, IBMQ
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.monitor import job_monitor
from operators.pauli_hamiltonian import PauliHamiltonian
from quantum_circuit.trotterization import trotter_electric, trotter_coupling, trotter_plaquette
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
from qiskit.providers.jobstatus import JobStatus
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from utils.matrix_operations import *
from utils.change_basis import *
from quantum_info.expectation import expectation
import matplotlib.pyplot as plt


sig = []
sig.append(np.eye(2))
sig.append(np.array([[0, 1], [1, 0]]))
sig.append(np.array([[0, -1j], [1j, 0]]))
sig.append(np.array([[1, 0], [0, -1]]))

        
def noise_mitigation_gate(circ_func, n_qubits, device_backend, input_mapping, shots = 1024, simulation = True):
    """
    Compute the process matrix of the noise mitigation gate
    
    Arguments:
    circ_func: function(qiskit.QuantumCircuit, input_mapping, unitary_sim), the function that constructs the target circuit structure on the given qiskit.QuantumCircuit
    n_qubits: int, the number of qubits
    device_backend: the backend obtained by provider.get_backend("ibmq_****")
    input_mapping: list of int, mapping of logical qubit -> physical qubit. input_mapping[lq] = pq.
    shots: int, the number of shots
    simulation: bool, True if the noise is estimated with QASM simulator. 
    """
    
    """ COMPUTE THE MATRIX SIGMA AND ITS INVERSE: SIGMA = Tr[P_k P_i P_l P_j]/d """
    SigTensor1 = np.zeros([4, 4, 4, 4], dtype=complex)
    SigTensor1[0, 0, 0, 0] = np.trace(matmul([sig[0], sig[0], sig[0], sig[0]]))/2
    SigTensor1[0, 1, 0, 0] = np.trace(matmul([sig[0], sig[0], sig[1], sig[0]]))/2
    SigTensor1[0, 2, 0, 0] = np.trace(matmul([sig[0], sig[0], sig[2], sig[0]]))/2
    SigTensor1[0, 3, 0, 0] = np.trace(matmul([sig[0], sig[0], sig[3], sig[0]]))/2
    SigTensor1[1, 0, 0, 0] = np.trace(matmul([sig[1], sig[0], sig[0], sig[0]]))/2
    SigTensor1[1, 1, 0, 0] = np.trace(matmul([sig[1], sig[0], sig[1], sig[0]]))/2
    SigTensor1[1, 2, 0, 0] = np.trace(matmul([sig[1], sig[0], sig[2], sig[0]]))/2
    SigTensor1[1, 3, 0, 0] = np.trace(matmul([sig[1], sig[0], sig[3], sig[0]]))/2
    SigTensor1[2, 0, 0, 0] = np.trace(matmul([sig[2], sig[0], sig[0], sig[0]]))/2
    SigTensor1[2, 1, 0, 0] = np.trace(matmul([sig[2], sig[0], sig[1], sig[0]]))/2
    SigTensor1[2, 2, 0, 0] = np.trace(matmul([sig[2], sig[0], sig[2], sig[0]]))/2
    SigTensor1[2, 3, 0, 0] = np.trace(matmul([sig[2], sig[0], sig[3], sig[0]]))/2
    SigTensor1[3, 0, 0, 0] = np.trace(matmul([sig[3], sig[0], sig[0], sig[0]]))/2
    SigTensor1[3, 1, 0, 0] = np.trace(matmul([sig[3], sig[0], sig[1], sig[0]]))/2
    SigTensor1[3, 2, 0, 0] = np.trace(matmul([sig[3], sig[0], sig[2], sig[0]]))/2
    SigTensor1[3, 3, 0, 0] = np.trace(matmul([sig[3], sig[0], sig[3], sig[0]]))/2
    SigTensor1[0, 0, 0, 1] = np.trace(matmul([sig[0], sig[0], sig[0], sig[1]]))/2
    SigTensor1[0, 1, 0, 1] = np.trace(matmul([sig[0], sig[0], sig[1], sig[1]]))/2
    SigTensor1[0, 2, 0, 1] = np.trace(matmul([sig[0], sig[0], sig[2], sig[1]]))/2
    SigTensor1[0, 3, 0, 1] = np.trace(matmul([sig[0], sig[0], sig[3], sig[1]]))/2
    SigTensor1[1, 0, 0, 1] = np.trace(matmul([sig[1], sig[0], sig[0], sig[1]]))/2
    SigTensor1[1, 1, 0, 1] = np.trace(matmul([sig[1], sig[0], sig[1], sig[1]]))/2
    SigTensor1[1, 2, 0, 1] = np.trace(matmul([sig[1], sig[0], sig[2], sig[1]]))/2
    SigTensor1[1, 3, 0, 1] = np.trace(matmul([sig[1], sig[0], sig[3], sig[1]]))/2
    SigTensor1[2, 0, 0, 1] = np.trace(matmul([sig[2], sig[0], sig[0], sig[1]]))/2
    SigTensor1[2, 1, 0, 1] = np.trace(matmul([sig[2], sig[0], sig[1], sig[1]]))/2
    SigTensor1[2, 2, 0, 1] = np.trace(matmul([sig[2], sig[0], sig[2], sig[1]]))/2
    SigTensor1[2, 3, 0, 1] = np.trace(matmul([sig[2], sig[0], sig[3], sig[1]]))/2
    SigTensor1[3, 0, 0, 1] = np.trace(matmul([sig[3], sig[0], sig[0], sig[1]]))/2
    SigTensor1[3, 1, 0, 1] = np.trace(matmul([sig[3], sig[0], sig[1], sig[1]]))/2
    SigTensor1[3, 2, 0, 1] = np.trace(matmul([sig[3], sig[0], sig[2], sig[1]]))/2
    SigTensor1[3, 3, 0, 1] = np.trace(matmul([sig[3], sig[0], sig[3], sig[1]]))/2
    SigTensor1[0, 0, 0, 2] = np.trace(matmul([sig[0], sig[0], sig[0], sig[2]]))/2
    SigTensor1[0, 1, 0, 2] = np.trace(matmul([sig[0], sig[0], sig[1], sig[2]]))/2
    SigTensor1[0, 2, 0, 2] = np.trace(matmul([sig[0], sig[0], sig[2], sig[2]]))/2
    SigTensor1[0, 3, 0, 2] = np.trace(matmul([sig[0], sig[0], sig[3], sig[2]]))/2
    SigTensor1[1, 0, 0, 2] = np.trace(matmul([sig[1], sig[0], sig[0], sig[2]]))/2
    SigTensor1[1, 1, 0, 2] = np.trace(matmul([sig[1], sig[0], sig[1], sig[2]]))/2
    SigTensor1[1, 2, 0, 2] = np.trace(matmul([sig[1], sig[0], sig[2], sig[2]]))/2
    SigTensor1[1, 3, 0, 2] = np.trace(matmul([sig[1], sig[0], sig[3], sig[2]]))/2
    SigTensor1[2, 0, 0, 2] = np.trace(matmul([sig[2], sig[0], sig[0], sig[2]]))/2
    SigTensor1[2, 1, 0, 2] = np.trace(matmul([sig[2], sig[0], sig[1], sig[2]]))/2
    SigTensor1[2, 2, 0, 2] = np.trace(matmul([sig[2], sig[0], sig[2], sig[2]]))/2
    SigTensor1[2, 3, 0, 2] = np.trace(matmul([sig[2], sig[0], sig[3], sig[2]]))/2
    SigTensor1[3, 0, 0, 2] = np.trace(matmul([sig[3], sig[0], sig[0], sig[2]]))/2
    SigTensor1[3, 1, 0, 2] = np.trace(matmul([sig[3], sig[0], sig[1], sig[2]]))/2
    SigTensor1[3, 2, 0, 2] = np.trace(matmul([sig[3], sig[0], sig[2], sig[2]]))/2
    SigTensor1[3, 3, 0, 2] = np.trace(matmul([sig[3], sig[0], sig[3], sig[2]]))/2
    SigTensor1[0, 0, 0, 3] = np.trace(matmul([sig[0], sig[0], sig[0], sig[3]]))/2
    SigTensor1[0, 1, 0, 3] = np.trace(matmul([sig[0], sig[0], sig[1], sig[3]]))/2
    SigTensor1[0, 2, 0, 3] = np.trace(matmul([sig[0], sig[0], sig[2], sig[3]]))/2
    SigTensor1[0, 3, 0, 3] = np.trace(matmul([sig[0], sig[0], sig[3], sig[3]]))/2
    SigTensor1[1, 0, 0, 3] = np.trace(matmul([sig[1], sig[0], sig[0], sig[3]]))/2
    SigTensor1[1, 1, 0, 3] = np.trace(matmul([sig[1], sig[0], sig[1], sig[3]]))/2
    SigTensor1[1, 2, 0, 3] = np.trace(matmul([sig[1], sig[0], sig[2], sig[3]]))/2
    SigTensor1[1, 3, 0, 3] = np.trace(matmul([sig[1], sig[0], sig[3], sig[3]]))/2
    SigTensor1[2, 0, 0, 3] = np.trace(matmul([sig[2], sig[0], sig[0], sig[3]]))/2
    SigTensor1[2, 1, 0, 3] = np.trace(matmul([sig[2], sig[0], sig[1], sig[3]]))/2
    SigTensor1[2, 2, 0, 3] = np.trace(matmul([sig[2], sig[0], sig[2], sig[3]]))/2
    SigTensor1[2, 3, 0, 3] = np.trace(matmul([sig[2], sig[0], sig[3], sig[3]]))/2
    SigTensor1[3, 0, 0, 3] = np.trace(matmul([sig[3], sig[0], sig[0], sig[3]]))/2
    SigTensor1[3, 1, 0, 3] = np.trace(matmul([sig[3], sig[0], sig[1], sig[3]]))/2
    SigTensor1[3, 2, 0, 3] = np.trace(matmul([sig[3], sig[0], sig[2], sig[3]]))/2
    SigTensor1[3, 3, 0, 3] = np.trace(matmul([sig[3], sig[0], sig[3], sig[3]]))/2
    SigTensor1[0, 0, 1, 0] = np.trace(matmul([sig[0], sig[1], sig[0], sig[0]]))/2
    SigTensor1[0, 1, 1, 0] = np.trace(matmul([sig[0], sig[1], sig[1], sig[0]]))/2
    SigTensor1[0, 2, 1, 0] = np.trace(matmul([sig[0], sig[1], sig[2], sig[0]]))/2
    SigTensor1[0, 3, 1, 0] = np.trace(matmul([sig[0], sig[1], sig[3], sig[0]]))/2
    SigTensor1[1, 0, 1, 0] = np.trace(matmul([sig[1], sig[1], sig[0], sig[0]]))/2
    SigTensor1[1, 1, 1, 0] = np.trace(matmul([sig[1], sig[1], sig[1], sig[0]]))/2
    SigTensor1[1, 2, 1, 0] = np.trace(matmul([sig[1], sig[1], sig[2], sig[0]]))/2
    SigTensor1[1, 3, 1, 0] = np.trace(matmul([sig[1], sig[1], sig[3], sig[0]]))/2
    SigTensor1[2, 0, 1, 0] = np.trace(matmul([sig[2], sig[1], sig[0], sig[0]]))/2
    SigTensor1[2, 1, 1, 0] = np.trace(matmul([sig[2], sig[1], sig[1], sig[0]]))/2
    SigTensor1[2, 2, 1, 0] = np.trace(matmul([sig[2], sig[1], sig[2], sig[0]]))/2
    SigTensor1[2, 3, 1, 0] = np.trace(matmul([sig[2], sig[1], sig[3], sig[0]]))/2
    SigTensor1[3, 0, 1, 0] = np.trace(matmul([sig[3], sig[1], sig[0], sig[0]]))/2
    SigTensor1[3, 1, 1, 0] = np.trace(matmul([sig[3], sig[1], sig[1], sig[0]]))/2
    SigTensor1[3, 2, 1, 0] = np.trace(matmul([sig[3], sig[1], sig[2], sig[0]]))/2
    SigTensor1[3, 3, 1, 0] = np.trace(matmul([sig[3], sig[1], sig[3], sig[0]]))/2
    SigTensor1[0, 0, 1, 1] = np.trace(matmul([sig[0], sig[1], sig[0], sig[1]]))/2
    SigTensor1[0, 1, 1, 1] = np.trace(matmul([sig[0], sig[1], sig[1], sig[1]]))/2
    SigTensor1[0, 2, 1, 1] = np.trace(matmul([sig[0], sig[1], sig[2], sig[1]]))/2
    SigTensor1[0, 3, 1, 1] = np.trace(matmul([sig[0], sig[1], sig[3], sig[1]]))/2
    SigTensor1[1, 0, 1, 1] = np.trace(matmul([sig[1], sig[1], sig[0], sig[1]]))/2
    SigTensor1[1, 1, 1, 1] = np.trace(matmul([sig[1], sig[1], sig[1], sig[1]]))/2
    SigTensor1[1, 2, 1, 1] = np.trace(matmul([sig[1], sig[1], sig[2], sig[1]]))/2
    SigTensor1[1, 3, 1, 1] = np.trace(matmul([sig[1], sig[1], sig[3], sig[1]]))/2
    SigTensor1[2, 0, 1, 1] = np.trace(matmul([sig[2], sig[1], sig[0], sig[1]]))/2
    SigTensor1[2, 1, 1, 1] = np.trace(matmul([sig[2], sig[1], sig[1], sig[1]]))/2
    SigTensor1[2, 2, 1, 1] = np.trace(matmul([sig[2], sig[1], sig[2], sig[1]]))/2
    SigTensor1[2, 3, 1, 1] = np.trace(matmul([sig[2], sig[1], sig[3], sig[1]]))/2
    SigTensor1[3, 0, 1, 1] = np.trace(matmul([sig[3], sig[1], sig[0], sig[1]]))/2
    SigTensor1[3, 1, 1, 1] = np.trace(matmul([sig[3], sig[1], sig[1], sig[1]]))/2
    SigTensor1[3, 2, 1, 1] = np.trace(matmul([sig[3], sig[1], sig[2], sig[1]]))/2
    SigTensor1[3, 3, 1, 1] = np.trace(matmul([sig[3], sig[1], sig[3], sig[1]]))/2
    SigTensor1[0, 0, 1, 2] = np.trace(matmul([sig[0], sig[1], sig[0], sig[2]]))/2
    SigTensor1[0, 1, 1, 2] = np.trace(matmul([sig[0], sig[1], sig[1], sig[2]]))/2
    SigTensor1[0, 2, 1, 2] = np.trace(matmul([sig[0], sig[1], sig[2], sig[2]]))/2
    SigTensor1[0, 3, 1, 2] = np.trace(matmul([sig[0], sig[1], sig[3], sig[2]]))/2
    SigTensor1[1, 0, 1, 2] = np.trace(matmul([sig[1], sig[1], sig[0], sig[2]]))/2
    SigTensor1[1, 1, 1, 2] = np.trace(matmul([sig[1], sig[1], sig[1], sig[2]]))/2
    SigTensor1[1, 2, 1, 2] = np.trace(matmul([sig[1], sig[1], sig[2], sig[2]]))/2
    SigTensor1[1, 3, 1, 2] = np.trace(matmul([sig[1], sig[1], sig[3], sig[2]]))/2
    SigTensor1[2, 0, 1, 2] = np.trace(matmul([sig[2], sig[1], sig[0], sig[2]]))/2
    SigTensor1[2, 1, 1, 2] = np.trace(matmul([sig[2], sig[1], sig[1], sig[2]]))/2
    SigTensor1[2, 2, 1, 2] = np.trace(matmul([sig[2], sig[1], sig[2], sig[2]]))/2
    SigTensor1[2, 3, 1, 2] = np.trace(matmul([sig[2], sig[1], sig[3], sig[2]]))/2
    SigTensor1[3, 0, 1, 2] = np.trace(matmul([sig[3], sig[1], sig[0], sig[2]]))/2
    SigTensor1[3, 1, 1, 2] = np.trace(matmul([sig[3], sig[1], sig[1], sig[2]]))/2
    SigTensor1[3, 2, 1, 2] = np.trace(matmul([sig[3], sig[1], sig[2], sig[2]]))/2
    SigTensor1[3, 3, 1, 2] = np.trace(matmul([sig[3], sig[1], sig[3], sig[2]]))/2
    SigTensor1[0, 0, 1, 3] = np.trace(matmul([sig[0], sig[1], sig[0], sig[3]]))/2
    SigTensor1[0, 1, 1, 3] = np.trace(matmul([sig[0], sig[1], sig[1], sig[3]]))/2
    SigTensor1[0, 2, 1, 3] = np.trace(matmul([sig[0], sig[1], sig[2], sig[3]]))/2
    SigTensor1[0, 3, 1, 3] = np.trace(matmul([sig[0], sig[1], sig[3], sig[3]]))/2
    SigTensor1[1, 0, 1, 3] = np.trace(matmul([sig[1], sig[1], sig[0], sig[3]]))/2
    SigTensor1[1, 1, 1, 3] = np.trace(matmul([sig[1], sig[1], sig[1], sig[3]]))/2
    SigTensor1[1, 2, 1, 3] = np.trace(matmul([sig[1], sig[1], sig[2], sig[3]]))/2
    SigTensor1[1, 3, 1, 3] = np.trace(matmul([sig[1], sig[1], sig[3], sig[3]]))/2
    SigTensor1[2, 0, 1, 3] = np.trace(matmul([sig[2], sig[1], sig[0], sig[3]]))/2
    SigTensor1[2, 1, 1, 3] = np.trace(matmul([sig[2], sig[1], sig[1], sig[3]]))/2
    SigTensor1[2, 2, 1, 3] = np.trace(matmul([sig[2], sig[1], sig[2], sig[3]]))/2
    SigTensor1[2, 3, 1, 3] = np.trace(matmul([sig[2], sig[1], sig[3], sig[3]]))/2
    SigTensor1[3, 0, 1, 3] = np.trace(matmul([sig[3], sig[1], sig[0], sig[3]]))/2
    SigTensor1[3, 1, 1, 3] = np.trace(matmul([sig[3], sig[1], sig[1], sig[3]]))/2
    SigTensor1[3, 2, 1, 3] = np.trace(matmul([sig[3], sig[1], sig[2], sig[3]]))/2
    SigTensor1[3, 3, 1, 3] = np.trace(matmul([sig[3], sig[1], sig[3], sig[3]]))/2
    SigTensor1[0, 0, 2, 0] = np.trace(matmul([sig[0], sig[2], sig[0], sig[0]]))/2
    SigTensor1[0, 1, 2, 0] = np.trace(matmul([sig[0], sig[2], sig[1], sig[0]]))/2
    SigTensor1[0, 2, 2, 0] = np.trace(matmul([sig[0], sig[2], sig[2], sig[0]]))/2
    SigTensor1[0, 3, 2, 0] = np.trace(matmul([sig[0], sig[2], sig[3], sig[0]]))/2
    SigTensor1[1, 0, 2, 0] = np.trace(matmul([sig[1], sig[2], sig[0], sig[0]]))/2
    SigTensor1[1, 1, 2, 0] = np.trace(matmul([sig[1], sig[2], sig[1], sig[0]]))/2
    SigTensor1[1, 2, 2, 0] = np.trace(matmul([sig[1], sig[2], sig[2], sig[0]]))/2
    SigTensor1[1, 3, 2, 0] = np.trace(matmul([sig[1], sig[2], sig[3], sig[0]]))/2
    SigTensor1[2, 0, 2, 0] = np.trace(matmul([sig[2], sig[2], sig[0], sig[0]]))/2
    SigTensor1[2, 1, 2, 0] = np.trace(matmul([sig[2], sig[2], sig[1], sig[0]]))/2
    SigTensor1[2, 2, 2, 0] = np.trace(matmul([sig[2], sig[2], sig[2], sig[0]]))/2
    SigTensor1[2, 3, 2, 0] = np.trace(matmul([sig[2], sig[2], sig[3], sig[0]]))/2
    SigTensor1[3, 0, 2, 0] = np.trace(matmul([sig[3], sig[2], sig[0], sig[0]]))/2
    SigTensor1[3, 1, 2, 0] = np.trace(matmul([sig[3], sig[2], sig[1], sig[0]]))/2
    SigTensor1[3, 2, 2, 0] = np.trace(matmul([sig[3], sig[2], sig[2], sig[0]]))/2
    SigTensor1[3, 3, 2, 0] = np.trace(matmul([sig[3], sig[2], sig[3], sig[0]]))/2
    SigTensor1[0, 0, 2, 1] = np.trace(matmul([sig[0], sig[2], sig[0], sig[1]]))/2
    SigTensor1[0, 1, 2, 1] = np.trace(matmul([sig[0], sig[2], sig[1], sig[1]]))/2
    SigTensor1[0, 2, 2, 1] = np.trace(matmul([sig[0], sig[2], sig[2], sig[1]]))/2
    SigTensor1[0, 3, 2, 1] = np.trace(matmul([sig[0], sig[2], sig[3], sig[1]]))/2
    SigTensor1[1, 0, 2, 1] = np.trace(matmul([sig[1], sig[2], sig[0], sig[1]]))/2
    SigTensor1[1, 1, 2, 1] = np.trace(matmul([sig[1], sig[2], sig[1], sig[1]]))/2
    SigTensor1[1, 2, 2, 1] = np.trace(matmul([sig[1], sig[2], sig[2], sig[1]]))/2
    SigTensor1[1, 3, 2, 1] = np.trace(matmul([sig[1], sig[2], sig[3], sig[1]]))/2
    SigTensor1[2, 0, 2, 1] = np.trace(matmul([sig[2], sig[2], sig[0], sig[1]]))/2
    SigTensor1[2, 1, 2, 1] = np.trace(matmul([sig[2], sig[2], sig[1], sig[1]]))/2
    SigTensor1[2, 2, 2, 1] = np.trace(matmul([sig[2], sig[2], sig[2], sig[1]]))/2
    SigTensor1[2, 3, 2, 1] = np.trace(matmul([sig[2], sig[2], sig[3], sig[1]]))/2
    SigTensor1[3, 0, 2, 1] = np.trace(matmul([sig[3], sig[2], sig[0], sig[1]]))/2
    SigTensor1[3, 1, 2, 1] = np.trace(matmul([sig[3], sig[2], sig[1], sig[1]]))/2
    SigTensor1[3, 2, 2, 1] = np.trace(matmul([sig[3], sig[2], sig[2], sig[1]]))/2
    SigTensor1[3, 3, 2, 1] = np.trace(matmul([sig[3], sig[2], sig[3], sig[1]]))/2
    SigTensor1[0, 0, 2, 2] = np.trace(matmul([sig[0], sig[2], sig[0], sig[2]]))/2
    SigTensor1[0, 1, 2, 2] = np.trace(matmul([sig[0], sig[2], sig[1], sig[2]]))/2
    SigTensor1[0, 2, 2, 2] = np.trace(matmul([sig[0], sig[2], sig[2], sig[2]]))/2
    SigTensor1[0, 3, 2, 2] = np.trace(matmul([sig[0], sig[2], sig[3], sig[2]]))/2
    SigTensor1[1, 0, 2, 2] = np.trace(matmul([sig[1], sig[2], sig[0], sig[2]]))/2
    SigTensor1[1, 1, 2, 2] = np.trace(matmul([sig[1], sig[2], sig[1], sig[2]]))/2
    SigTensor1[1, 2, 2, 2] = np.trace(matmul([sig[1], sig[2], sig[2], sig[2]]))/2
    SigTensor1[1, 3, 2, 2] = np.trace(matmul([sig[1], sig[2], sig[3], sig[2]]))/2
    SigTensor1[2, 0, 2, 2] = np.trace(matmul([sig[2], sig[2], sig[0], sig[2]]))/2
    SigTensor1[2, 1, 2, 2] = np.trace(matmul([sig[2], sig[2], sig[1], sig[2]]))/2
    SigTensor1[2, 2, 2, 2] = np.trace(matmul([sig[2], sig[2], sig[2], sig[2]]))/2
    SigTensor1[2, 3, 2, 2] = np.trace(matmul([sig[2], sig[2], sig[3], sig[2]]))/2
    SigTensor1[3, 0, 2, 2] = np.trace(matmul([sig[3], sig[2], sig[0], sig[2]]))/2
    SigTensor1[3, 1, 2, 2] = np.trace(matmul([sig[3], sig[2], sig[1], sig[2]]))/2
    SigTensor1[3, 2, 2, 2] = np.trace(matmul([sig[3], sig[2], sig[2], sig[2]]))/2
    SigTensor1[3, 3, 2, 2] = np.trace(matmul([sig[3], sig[2], sig[3], sig[2]]))/2
    SigTensor1[0, 0, 2, 3] = np.trace(matmul([sig[0], sig[2], sig[0], sig[3]]))/2
    SigTensor1[0, 1, 2, 3] = np.trace(matmul([sig[0], sig[2], sig[1], sig[3]]))/2
    SigTensor1[0, 2, 2, 3] = np.trace(matmul([sig[0], sig[2], sig[2], sig[3]]))/2
    SigTensor1[0, 3, 2, 3] = np.trace(matmul([sig[0], sig[2], sig[3], sig[3]]))/2
    SigTensor1[1, 0, 2, 3] = np.trace(matmul([sig[1], sig[2], sig[0], sig[3]]))/2
    SigTensor1[1, 1, 2, 3] = np.trace(matmul([sig[1], sig[2], sig[1], sig[3]]))/2
    SigTensor1[1, 2, 2, 3] = np.trace(matmul([sig[1], sig[2], sig[2], sig[3]]))/2
    SigTensor1[1, 3, 2, 3] = np.trace(matmul([sig[1], sig[2], sig[3], sig[3]]))/2
    SigTensor1[2, 0, 2, 3] = np.trace(matmul([sig[2], sig[2], sig[0], sig[3]]))/2
    SigTensor1[2, 1, 2, 3] = np.trace(matmul([sig[2], sig[2], sig[1], sig[3]]))/2
    SigTensor1[2, 2, 2, 3] = np.trace(matmul([sig[2], sig[2], sig[2], sig[3]]))/2
    SigTensor1[2, 3, 2, 3] = np.trace(matmul([sig[2], sig[2], sig[3], sig[3]]))/2
    SigTensor1[3, 0, 2, 3] = np.trace(matmul([sig[3], sig[2], sig[0], sig[3]]))/2
    SigTensor1[3, 1, 2, 3] = np.trace(matmul([sig[3], sig[2], sig[1], sig[3]]))/2
    SigTensor1[3, 2, 2, 3] = np.trace(matmul([sig[3], sig[2], sig[2], sig[3]]))/2
    SigTensor1[3, 3, 2, 3] = np.trace(matmul([sig[3], sig[2], sig[3], sig[3]]))/2
    SigTensor1[0, 0, 3, 0] = np.trace(matmul([sig[0], sig[3], sig[0], sig[0]]))/2
    SigTensor1[0, 1, 3, 0] = np.trace(matmul([sig[0], sig[3], sig[1], sig[0]]))/2
    SigTensor1[0, 2, 3, 0] = np.trace(matmul([sig[0], sig[3], sig[2], sig[0]]))/2
    SigTensor1[0, 3, 3, 0] = np.trace(matmul([sig[0], sig[3], sig[3], sig[0]]))/2
    SigTensor1[1, 0, 3, 0] = np.trace(matmul([sig[1], sig[3], sig[0], sig[0]]))/2
    SigTensor1[1, 1, 3, 0] = np.trace(matmul([sig[1], sig[3], sig[1], sig[0]]))/2
    SigTensor1[1, 2, 3, 0] = np.trace(matmul([sig[1], sig[3], sig[2], sig[0]]))/2
    SigTensor1[1, 3, 3, 0] = np.trace(matmul([sig[1], sig[3], sig[3], sig[0]]))/2
    SigTensor1[2, 0, 3, 0] = np.trace(matmul([sig[2], sig[3], sig[0], sig[0]]))/2
    SigTensor1[2, 1, 3, 0] = np.trace(matmul([sig[2], sig[3], sig[1], sig[0]]))/2
    SigTensor1[2, 2, 3, 0] = np.trace(matmul([sig[2], sig[3], sig[2], sig[0]]))/2
    SigTensor1[2, 3, 3, 0] = np.trace(matmul([sig[2], sig[3], sig[3], sig[0]]))/2
    SigTensor1[3, 0, 3, 0] = np.trace(matmul([sig[3], sig[3], sig[0], sig[0]]))/2
    SigTensor1[3, 1, 3, 0] = np.trace(matmul([sig[3], sig[3], sig[1], sig[0]]))/2
    SigTensor1[3, 2, 3, 0] = np.trace(matmul([sig[3], sig[3], sig[2], sig[0]]))/2
    SigTensor1[3, 3, 3, 0] = np.trace(matmul([sig[3], sig[3], sig[3], sig[0]]))/2
    SigTensor1[0, 0, 3, 1] = np.trace(matmul([sig[0], sig[3], sig[0], sig[1]]))/2
    SigTensor1[0, 1, 3, 1] = np.trace(matmul([sig[0], sig[3], sig[1], sig[1]]))/2
    SigTensor1[0, 2, 3, 1] = np.trace(matmul([sig[0], sig[3], sig[2], sig[1]]))/2
    SigTensor1[0, 3, 3, 1] = np.trace(matmul([sig[0], sig[3], sig[3], sig[1]]))/2
    SigTensor1[1, 0, 3, 1] = np.trace(matmul([sig[1], sig[3], sig[0], sig[1]]))/2
    SigTensor1[1, 1, 3, 1] = np.trace(matmul([sig[1], sig[3], sig[1], sig[1]]))/2
    SigTensor1[1, 2, 3, 1] = np.trace(matmul([sig[1], sig[3], sig[2], sig[1]]))/2
    SigTensor1[1, 3, 3, 1] = np.trace(matmul([sig[1], sig[3], sig[3], sig[1]]))/2
    SigTensor1[2, 0, 3, 1] = np.trace(matmul([sig[2], sig[3], sig[0], sig[1]]))/2
    SigTensor1[2, 1, 3, 1] = np.trace(matmul([sig[2], sig[3], sig[1], sig[1]]))/2
    SigTensor1[2, 2, 3, 1] = np.trace(matmul([sig[2], sig[3], sig[2], sig[1]]))/2
    SigTensor1[2, 3, 3, 1] = np.trace(matmul([sig[2], sig[3], sig[3], sig[1]]))/2
    SigTensor1[3, 0, 3, 1] = np.trace(matmul([sig[3], sig[3], sig[0], sig[1]]))/2
    SigTensor1[3, 1, 3, 1] = np.trace(matmul([sig[3], sig[3], sig[1], sig[1]]))/2
    SigTensor1[3, 2, 3, 1] = np.trace(matmul([sig[3], sig[3], sig[2], sig[1]]))/2
    SigTensor1[3, 3, 3, 1] = np.trace(matmul([sig[3], sig[3], sig[3], sig[1]]))/2
    SigTensor1[0, 0, 3, 2] = np.trace(matmul([sig[0], sig[3], sig[0], sig[2]]))/2
    SigTensor1[0, 1, 3, 2] = np.trace(matmul([sig[0], sig[3], sig[1], sig[2]]))/2
    SigTensor1[0, 2, 3, 2] = np.trace(matmul([sig[0], sig[3], sig[2], sig[2]]))/2
    SigTensor1[0, 3, 3, 2] = np.trace(matmul([sig[0], sig[3], sig[3], sig[2]]))/2
    SigTensor1[1, 0, 3, 2] = np.trace(matmul([sig[1], sig[3], sig[0], sig[2]]))/2
    SigTensor1[1, 1, 3, 2] = np.trace(matmul([sig[1], sig[3], sig[1], sig[2]]))/2
    SigTensor1[1, 2, 3, 2] = np.trace(matmul([sig[1], sig[3], sig[2], sig[2]]))/2
    SigTensor1[1, 3, 3, 2] = np.trace(matmul([sig[1], sig[3], sig[3], sig[2]]))/2
    SigTensor1[2, 0, 3, 2] = np.trace(matmul([sig[2], sig[3], sig[0], sig[2]]))/2
    SigTensor1[2, 1, 3, 2] = np.trace(matmul([sig[2], sig[3], sig[1], sig[2]]))/2
    SigTensor1[2, 2, 3, 2] = np.trace(matmul([sig[2], sig[3], sig[2], sig[2]]))/2
    SigTensor1[2, 3, 3, 2] = np.trace(matmul([sig[2], sig[3], sig[3], sig[2]]))/2
    SigTensor1[3, 0, 3, 2] = np.trace(matmul([sig[3], sig[3], sig[0], sig[2]]))/2
    SigTensor1[3, 1, 3, 2] = np.trace(matmul([sig[3], sig[3], sig[1], sig[2]]))/2
    SigTensor1[3, 2, 3, 2] = np.trace(matmul([sig[3], sig[3], sig[2], sig[2]]))/2
    SigTensor1[3, 3, 3, 2] = np.trace(matmul([sig[3], sig[3], sig[3], sig[2]]))/2
    SigTensor1[0, 0, 3, 3] = np.trace(matmul([sig[0], sig[3], sig[0], sig[3]]))/2
    SigTensor1[0, 1, 3, 3] = np.trace(matmul([sig[0], sig[3], sig[1], sig[3]]))/2
    SigTensor1[0, 2, 3, 3] = np.trace(matmul([sig[0], sig[3], sig[2], sig[3]]))/2
    SigTensor1[0, 3, 3, 3] = np.trace(matmul([sig[0], sig[3], sig[3], sig[3]]))/2
    SigTensor1[1, 0, 3, 3] = np.trace(matmul([sig[1], sig[3], sig[0], sig[3]]))/2
    SigTensor1[1, 1, 3, 3] = np.trace(matmul([sig[1], sig[3], sig[1], sig[3]]))/2
    SigTensor1[1, 2, 3, 3] = np.trace(matmul([sig[1], sig[3], sig[2], sig[3]]))/2
    SigTensor1[1, 3, 3, 3] = np.trace(matmul([sig[1], sig[3], sig[3], sig[3]]))/2
    SigTensor1[2, 0, 3, 3] = np.trace(matmul([sig[2], sig[3], sig[0], sig[3]]))/2
    SigTensor1[2, 1, 3, 3] = np.trace(matmul([sig[2], sig[3], sig[1], sig[3]]))/2
    SigTensor1[2, 2, 3, 3] = np.trace(matmul([sig[2], sig[3], sig[2], sig[3]]))/2
    SigTensor1[2, 3, 3, 3] = np.trace(matmul([sig[2], sig[3], sig[3], sig[3]]))/2
    SigTensor1[3, 0, 3, 3] = np.trace(matmul([sig[3], sig[3], sig[0], sig[3]]))/2
    SigTensor1[3, 1, 3, 3] = np.trace(matmul([sig[3], sig[3], sig[1], sig[3]]))/2
    SigTensor1[3, 2, 3, 3] = np.trace(matmul([sig[3], sig[3], sig[2], sig[3]]))/2
    SigTensor1[3, 3, 3, 3] = np.trace(matmul([sig[3], sig[3], sig[3], sig[3]]))/2
    SigMat = np.reshape(tensor_prod([SigTensor1 for i in range(n_qubits)]), [4**(2*n_qubits), 4**(2*n_qubits)])
    #SigMat_H = np.conjugate(SigMat).T 
                   
    """ COMPUTE THE REFERENCE STATES: rho_i """
    # Compute the 1-qubit states that are bases of X, Y, Z
    ref1 = [
        np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex),
        np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex),
        np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex),
        np.array([1, 0], dtype=complex),
        np.array([0, 1], dtype=complex)
    ]
    
    # Compute the n_qubit reference states
    ref = []
    for i, inds_i in enumerate([p for p in itertools.product(list(range(6)), repeat=n_qubits)]):
        state = tensor_prod([ref1[ind] for ind in inds_i])
        ref.append(np.outer(np.conjugate(state), state))
        
    
    """ COMPUTE THE MATRIX S """
    backend = Aer.get_backend('unitary_simulator')
    cr = ClassicalRegister(n_qubits, 'cr')
    qr = QuantumRegister(n_qubits, 'qr')
    circ = QuantumCircuit(qr, cr)
    circ = circ_func(circ, list(range(n_qubits)), True)
    job = execute(circ, backend)
    U = job.result().get_unitary()
    S = np.zeros((6**n_qubits, 4**n_qubits), complex)
    for k, inds_k in enumerate([p for p in itertools.product(list(range(4)), repeat=n_qubits)]):
        for i in range(6**n_qubits):
            S[i, k] = np.trace(matmul([tensor_prod([sig[ind] for ind in inds_k]), U, ref[i], np.conjugate(U).T]))/2**(n_qubits/2)
    S_H = np.conjugate(S).T
    SS = np.matmul(S_H, S)
    SS_inv = np.linalg.inv(SS)
    
    
    """ COMPUTE THE PTM OF EPSILON (NOISE) """
    if simulation: 
        noise_model = NoiseModel.from_backend(device_backend)
        coupling_map = device_backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
        backend = Aer.get_backend('qasm_simulator')
        
    job_manager = IBMQJobManager()
    circs = []
    
    P_S = np.zeros((6**n_qubits, 4**n_qubits))
    for i, inds_i in enumerate([p for p in itertools.product(list(range(6)), repeat=n_qubits)]): 
        # the input state rho_i = |i1>|i2> where |i1>, |i2> \in {|0>, |1>, |+>, |->, |R>, |L>}
        for j, inds_j in enumerate([p for p in itertools.product(list(range(4)), repeat=n_qubits)]):
            if j == 0:
                P_S[i, j] = 1.0
                continue
            else:
                cr = ClassicalRegister(n_qubits, 'cr')
                if simulation:
                    qr = QuantumRegister(n_qubits, 'qr')
                    circ = QuantumCircuit(qr, cr)
                else:
                    circ = QuantumCircuit(device_backend.configuration().n_qubits, n_qubits)
                for trg_i, ind_i in enumerate(inds_i):
                    change_basis_input(circ, ind_i, input_mapping[n_qubits-trg_i-1])
                circ = circ_func(circ, input_mapping, False)
                measurement_trg_list = []
                for trg_j, ind_j in enumerate(inds_j):
                    if ind_j != 0:
                        # Specify to measure the trg_j-th qubit with the basis specified with j
                        change_basis_measurement(circ, ind_j, input_mapping[n_qubits-trg_j-1])
                        measurement_trg_list.append(input_mapping[n_qubits-trg_j-1])
                circ.measure(measurement_trg_list, list(range(len(measurement_trg_list))))
                if simulation: 
                    job = execute(circ, backend, 
                           noise_model=noise_model,
                           coupling_map=coupling_map,
                           basis_gates=basis_gates,
                           shots = shots)
                    P_S[i, j] =expectation(job.result().get_counts(circ), shots)
                else:
                    #job = execute(circ, device_backend, shots=shots)
                    circs.append(circ)
    #fig, ax = plt.subplots()
    #circs[58].draw(output='mpl', ax=ax)
    #plt.show()
    if not simulation: # Run multiple experiments on the device
        jobs = job_manager.run(circs, backend=device_backend)
        print(jobs.report())
        while any(stat != JobStatus.DONE for stat in jobs.statuses()):
            time.sleep(60)
        results = jobs.results()
        print(jobs.error_messages())
        for i in range(6**n_qubits):
            for j in range(1, 4**n_qubits):
                P_S[i, j] = expectation(results.get_counts(i*(4**n_qubits-1)+j-1), shots)
    P_S /= 2**(n_qubits/2)
    ptm_eps = matmul([SS_inv, S_H, P_S]).T
    
    
    ptm_eps_inv = np.linalg.inv(ptm_eps)
    
    """ COMPUTE THE PROCESS MATRIX OF EPSILON^-1"""
    ptm_eps_inv_vec = np.reshape(ptm_eps_inv, [-1, 1])
    #chi_eps_inv_diag = np.dot(np.matmul(np.linalg.inv(np.matmul(SigMat_H, SigMat)), SigMat_H), ptm_eps_inv_vec)
    chi_eps_inv = np.dot(np.linalg.inv(SigMat), ptm_eps_inv_vec)
    
    return chi_eps_inv
    