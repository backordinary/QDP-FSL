# https://github.com/JorgeAGR/nmsu-course-work/blob/6cd204abbc074734fb7e8ca0e693a15e1cbe4ede/PHYS520/Project/project_test.py
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import math
import random
import numpy as np
from scipy.optimize import minimize
#%config InlineBackend.figure_format = 'svg' # Makes the images look nice
backend = Aer.get_backend('qasm_simulator')
shots = 200000

A = np.array([[2, 1],
              [4.3, 1],
              [5.8, 1],
              [8.6, 1]])

b = np.array([[4.2],
              [8.5],
              [11.3],
              [15.9]])

norm_b = np.linalg.norm(A.T@b)
# Need to normalize b
ketb = (A.T @ b) / norm_b
A_h = A.T @ A / norm_b
# Follows the order A_h = cI + cZ + cX
c_l = [(A_h[0][0] + A_h[1][1])/2, (A_h[0][0] - A_h[1][1])/2, A_h[0][1]]
# The phase to rotate the vector by
b_rotation = np.arccos(ketb[0][0])

x = np.linalg.inv(A.T @ A) @ A.T @ b
x_norm = np.linalg.norm(x)
x_c = x / x_norm
print('Classical Solution: {}'.format(x_c.tolist()))
print('Optimal Alpha: {:.4f}\n'.format(np.arccos(x_c[0,0])*2))

n_gates = 3
n_qubits = 1
tot_qubits = n_qubits + 1 # qubits for model size + 1 ancillary qbit
aux_id = n_qubits # index for the ancillary qbit

# def A_l(index):
#     if index == 0:
#         # Identity
#         None
#     elif index == 1:
#         qc.cz(qr[aux_id], qr[0])
#     elif index == 2:
#         qc.cx(qr[aux_id], qr[0])

# # Rx will prepare |0> to the state |b> needed...?
# def U(phase):
#     qc.ry(phase*2, qr[0])

# # Controlled U
# def CU(phase):
#     qc.cry(phase*2, qr[aux_id], qr[0])

# # Fixed Ansatz
# def V(params):
#     for i in range(n_qubits):
#         #qc.h(qr[i])
#         qc.ry(params[i], qr[i])

# # Controlled Fixed Ansatz
# def CV(params):
#     for i in range(n_qubits):
#         #qc.ch(qr[aux_id], qr[i])
#         qc.cry(params[i], qr[aux_id], qr[i])

# def hadamard_Test_Beta(params, l, lp):
#     qc.h(qr[aux_id])
#     V(params)
#     A_l(l)
#     A_l(lp)
#     qc.h(qr[aux_id])
#     qc.measure(qr[aux_id], cr[0])
#     job = execute(qc, backend, shots=shots)
#     results = job.result().get_counts()
#     if '1' in results.keys():
#         beta = 1 - 2*job.result().get_counts()['1']/shots
#     else:
#         beta = 1
#     # Uncompute to reset qubits
#     for q in range(tot_qubits):
#         qc.reset(qr[q])
#     return beta

# def hadamard_Test_Gamma(params, l, reverse=0):
#     qc.h(qr[aux_id])
#     if not reverse:
#         V(params)
#         A_l(l)
#         CU(-phase)
#     else:
#         U(-phase)
#         A_l(l)
#         CV(params)
#     qc.h(qr[aux_id])

def local_Hadamard_Test(alpha, ancilla, qubit, b_rotation, l, lp, j, real=1):
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    qc.h(ancilla)
    
    if real == 0:
        qc.sdg(ancilla)
    
    # Ansatz
    #qc.h(qubit)
    qc.ry(alpha, qubit)
    
    # Data unitaries A_l
    if (l == 1):
        qc.cz(ancilla, qubit)
    elif (l == 2):
        qc.cx(ancilla, qubit)
    
    if j != -1:
        #U dagger
        qc.ry(-2*b_rotation, qubit)
        qc.cz(ancilla, qubit)
        #U
        qc.ry(2*b_rotation, qubit)
    
    # Data unitaries A_l'
    if (lp == 1):
        qc.cz(ancilla, qubit)
    elif (lp == 2):
        qc.cx(ancilla, qubit)
    
    qc.h(ancilla)
    
    # backend = Aer.get_backend('statevector_simulator')

    # job = execute(qc, backend)
    
    # result = job.result()
    # print(result.get_statevector())
    
    qc.measure(ancilla, 0)
    
    job = execute(qc, backend, shots=shots)

    result = job.result().get_counts()
    
    if ('1' in result.keys()):
        p1 = result['1']/shots
    else:
        p1 = 0
    
    return 1-2*p1

def mu(alpha, b_rotation, l, lp, j):
    ancilla = 0
    qubit = 1
    
    mu_r = local_Hadamard_Test(alpha, ancilla, qubit, b_rotation, l, lp, j)
    #mu_i = local_Hadamard_Test(alpha, ancilla, qubit, b_rotation, l, lp, j, 0)
    
    return mu_r# + mu_i*1j

def psi_Norm(alpha, b_rotation):
    norm = 0
    
    for l in range(len(c_l)):
        for lp in range(len(c_l)):
            norm = norm + c_l[l] * c_l[lp] * mu(alpha, b_rotation, l, lp, -1)
    
    return np.abs(norm)

def local_Cost(alpha):
    mu_sum = 0
    
    for l in range(len(c_l)):
        for lp in range(len(c_l)):
            mu_sum = mu_sum + c_l[l] * c_l[lp] * mu(alpha, b_rotation, l, lp, 1)
    
    norm = psi_Norm(alpha, b_rotation)
    mu_sum = np.abs(mu_sum)
    print('{}: {}'.format(alpha, 0.5 - 0.5 * mu_sum / norm))
    return 0.5 - 0.5 * mu_sum / norm

# def eval_Cost(params):
#     psipsi = 0
#     bpsi_sq = 0
#     for l in range(n_gates):
#         for lp in range(n_gates):
#             beta = hadamard_Test_Beta(params, l, lp)
#             #print(beta)
#             psipsi += c_l[l]*c_l[lp]*beta
            
#             gamma = 1   
#             for j, k in enumerate((l, lp)):
#                 hadamard_Test_Gamma(params, k)
#                 qc.measure(qr[aux_id], cr[0])
#                 job = execute(qc, backend, shots=shots)
#                 results = job.result().get_counts()
#                 if '1' in results.keys():
#                     gamma = gamma * (1 - 2*job.result().get_counts()['1']/shots)
#                 else:
#                     gamma = gamma * 1
#                 bpsi_sq += c_l[l]*c_l[lp]*gamma
#                 for q in range(tot_qubits):
#                     qc.reset(qr[q])
#     print(1 - (bpsi_sq / psipsi))
#     return 1 - (bpsi_sq / psipsi)

alphas = np.arange(0.6, 1, 0.1)
costs = np.zeros(alphas.shape)
for i, a in enumerate(alphas):
    costs[i] = local_Cost(a)

#opt = minimize(local_Cost, x0=np.random.rand(), method='COBYLA',
#                options={'maxiter':50})
'''
qr = QuantumRegister(tot_qubits)
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)
shots = 10000

opt = minimize(eval_Cost, x0=[0,], method='L-BFGS-B',
                options={'maxiter':200})
'''
# hadamard_Test_Beta(np.random.rand(1), 0, 0)
# eval_Cost([1,])
# qc.measure(qr[0], cr[0])
# job = execute(qc, backend, shots=10000, memory=True)#noise_model=noise_model,
#                                         # coupling_map=coupling_map,
#                                         # basis_gates=noise_model.basis_gates)
# output = job.result().get_counts()
# print(output)
# qc.draw('mpl')
