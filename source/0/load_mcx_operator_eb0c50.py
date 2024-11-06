# https://github.com/psasanka1729/Cluster-runs/blob/98270dc46a0e91aa88183a0d2706f32a11b2b067/load_MCX_operator.py
'''

This code takes a MCX gate and number of qubits as input and returns
the Grover operator.

'''


import time
import numpy as np
from scipy.sparse import identity
from scipy import sparse
from scipy.sparse import lil_matrix
import scipy.sparse.linalg
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
import sys
np_load_old = np.load

# Loads the MCX operator.
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k, encoding='bytes')

N = 8

B = np.load('0.0_U0_operator.npy')

MCX = np.matrix(B)



# The operator Ux.
A = np.ones((2**N, 2**N))
U_x = (2/(2**N))*A - np.identity(2**N, dtype = complex)
U_x_sp = sparse.csr_matrix(U_x)

## The operator U_0. This is neeed for the sign adjustment of Grover_reconstructed operator.
U_0 = - np.identity(2 ** N, dtype=complex) 
Target_state = '0'*N
Target_index = int(Target_state, 2)
U_0.itemset((Target_index, Target_index),1)



print(np.around(B,2))