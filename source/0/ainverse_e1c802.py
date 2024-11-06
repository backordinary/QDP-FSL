# https://github.com/ugur42/Qiskit-CKS-Algorithm/blob/a45a2c3875766195360ef08ed86c22d35c0d735d/MatrixInverse/Ainverse.py

"""
Implementing the inversion of A with Hamiltonmian SImulation
(Fourier approach in CKS paper, see red box1 Page 5)
"""


from qiskit import *
import numpy as np
from qiskit.visualization import plot_histogram
import qiskit.tools.jupyter
import matplotlib.pyplot as plt
from qiskit.aqua.algorithms import IQPE
from scipy.linalg import expm, sinm, cosm #for the bridge Hamiltonian simulation
from qiskit.extensions import *

from qiskit.quantum_info.operators import Operator

A = np.array([[1/2+0.j,-1/3],[-1/3,1/2]])
# A =A/np.linalg.norm(A)
A_inv = qiskit.extensions.HamiltonianGate(A, 1)
A_inv1 = A_inv.to_matrix()
numpInv=np.linalg.inv(A)
print(np.dot(A,A_inv1))
