# https://github.com/vaap96/sym_breaking/blob/6a52d51b3d2263fe9dc3f744e26d06238e95284a/gen_functions.py
import numpy as np
import sys
import scipy.linalg as la
import matplotlib.pyplot as plt
import random
import qiskit
from scipy.optimize import minimize
from matplotlib import colors
import math
import copy

# Define Pauli Matrices
pauli_x = np.array([[0,1],[1,0]])
pauli_y = np.array([[0,-1j],[1j,0]])
pauli_z = np.array([[1,0],[0,-1]])
identity = np.array([[1,0],[0,1]])

# Initial Parameters used for the optimization procedure
theta = 0
phi = 0
ket_0 = np.array([[1],[0]])
ket_1 = np.array([[0],[1]])
bra_0 = ket_0.transpose()
bra_1 = ket_1.transpose()

# Function for the Kronecker product 
def kronecker(pos, qubits, pauli):    
    if pos == 0:
        tmp = pauli
    else:
        tmp = identity
    for i in range(1, qubits):        
        if i != pos:
            tmp = np.kron(tmp, identity)
        else:
            tmp = np.kron(tmp, pauli)
        
    return tmp

# Dictionary of the Kronecker Products between the Pauli Matrices
def g_rep(pauli, qubits):
    pauli_dict = {}
    for i in range(qubits):
        pauli_dict[i] = kronecker(i, qubits, pauli)
    
    return pauli_dict

# Representation of the Hamiltonian (Either periodic or open boundaries)
def ham_rep(htype, qubits, delta, jy, jz):
    dict_X = g_rep(pauli_x, qubits)
    dict_Y = g_rep(pauli_y, qubits)
    dict_Z = g_rep(pauli_z, qubits)
    if htype == 'ob':
        hlistX = []
        hlistY = []
        hlistZ = []
        for i in range(qubits - 1):
            hlistX.append(np.dot(dict_X[i], dict_X[i+1]))
        for i in range(qubits - 1):
            hlistY.append(np.dot(dict_Y[i], dict_Y[i+1]))
        for i in range(qubits - 1):
            hlistZ.append(np.dot(dict_Z[i], dict_Z[i+1]))
        
        ham_rep = delta * sum(hlistX) + jy * sum(hlistY) + jz * sum(hlistZ)
        
        return ham_rep

    if htype == 'pb':
        hlistX = []
        hlistY = []
        hlistZ = []
        for i in range(qubits - 1):
            hlistX.append(np.dot(dict_X[i], dict_X[i+1]))
        hlistX.append(np.dot(dict_X[qubits - 1], dict_X[0]))
        for i in range(qubits - 1):
            hlistY.append(np.dot(dict_Y[i], dict_Y[i+1]))
        hlistY.append(np.dot(dict_Y[qubits - 1], dict_Y[0]))
        for i in range(qubits - 1):
            hlistZ.append(np.dot(dict_Z[i], dict_Z[i+1]))
        hlistZ.append(np.dot(dict_Z[qubits - 1], dict_Z[0]))
        
        ham_rep = delta * sum(hlistX) + sum(hlistY) + sum(hlistZ)
        
        return ham_rep

# Create initial vectors and parameters
def vector_rep(qubits_no, param):
    psi_ket = []
    for i in range(qubits_no):
        if i == 0:
            psi_ket.append(np.cos(param[i]) * ket_0 + np.sin(param[i]) * np.exp(-1j * param[i+1]) * ket_1)
        else:
            psi_ket.append(np.cos(param[i+1]) * ket_0 + np.sin(param[i+1]) * np.exp(-1j * param[i+2]) * ket_1)

    tmp = psi_ket[0]
    for i in range(qubits_no-1):
        tmp = np.kron(tmp, psi_ket[i+1])

    return tmp, np.linalg.norm(tmp)

vec = vector_rep(2, [np.pi / 2,0,0,0])

# Optimize the parameters of the Hamiltonians
def ham_opt(param, ham, qubits):
    vector = vector_rep(qubits, param)[0]

    return np.dot(np.dot(np.conj(vector).transpose(), ham), vector)[0,0]

# Get results for 100 optimizations runs
def opt_res(ham):
   results = []
   for _ in range(100):
       res = minimize(ham_opt, np.random.uniform(-np.pi,np.pi, size=4), 
                  args = (ham, 2), method="COBYLA", options={"maxiter":5000})
       results.append(list(res.x))

   return results 

# Plot the points in the Bloch Sphere
def bloch_plt(results, filename):
   b = qutip.Bloch()
   for res in results:
       x = [np.sin(2*res[0])*np.cos(res[1]), np.sin(2*res[2])*np.cos(res[3])]
       y = [np.sin(2*res[0])*np.sin(res[1]), np.sin(2*res[2])*np.sin(res[3])]
       z = [np.cos(2*res[0]), np.cos(2*res[2])]

       p = [x,y,z]
       b.add_points(p)
   b.save(filename)

# Plot the eigvals of the open and periodic boundaries of the Hamiltonians
def eigval_plt(eigvals, filename):
   plt.plot(sorted(eigvals))
   plt.savefig(filename)
   plt.close()

# Matrix plot for showing the block structure
def matrix_plt(matrix, filename):
	norm = colors.LogNorm(clip = True)
	#my_cmap = copy.copy(plt.cm.get_cmap('viridis'))
	#my_cmap.set_bad((0,0,0))
	#plt.imshow(matrix, norm=norm, interpolation='nearest', cmap=my_cmap)
	plt.imshow(np.abs(matrix), norm=norm)
	plt.colorbar()
	plt.savefig(filename)
	plt.close()

# Find the projectors of group generators
def projectors(group, qubits):
    eigvals, eigvecs = la.eigh(group)

    eigvals_r = eigvals.round(2)
    eigvecs_r = eigvecs.round(2)

    vecs = []
    for i in range(len(eigvecs_r)):
        vecs.append(eigvecs_r[:, i])

    dict_c = {}
    for l in eigvals_r:
        if l not in dict_c:
            dict_c[l] = np.zeros((2 ** qubits, 2 ** qubits), dtype=np.complex_)

    for i, vec in enumerate(vecs):
        for k in dict_c.keys():
            if k == eigvals_r[i]:
                dict_c[eigvals_r[i]] += np.outer(vec, vec.conj())

    return dict_c
