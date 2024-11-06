# https://github.com/bram859/Quantum-Engineering-Group-Project/blob/041267fecbbfca96f5b1880a682e29cf8bfd7344/HHL.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random 
import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.algorithms.linear_solvers.hhl import HHL
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


N = 8
a = 1
b = -1/3
matrix = np.zeros((N,N),dtype=complex)
for j in range(N):
    for i in range(N):
        if j == i:
            matrix[j][i] = a
        elif i+1 == j:
            matrix[j][i] = b
        elif i-1 == j:
            matrix[j][i] = b

vector = np.zeros(N)
x =  random.randint(0, N-1)
print(x)
vector[x] = 1
scaling = np.linalg.norm(vector)
#vector = vector/scaling 

print(matrix)
naive_hhl_solution = [0, 0, 0, 0, 0, 0, 0, 0]
naive_hhl_solution[0] = HHL().solve(matrix, vector)
for j in range(N):
    vector[j] = 1
    naive_hhl_solution[j] = HHL().solve(matrix, vector)
    vector[j] = 0
    break
#making solutions for all the pure output states 

print(naive_hhl_solution.state())


# In[16]:


#classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
print(naive_hhl_solution)
print(tridi_solution)


# In[15]:


from qiskit.algorithms.linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
# doing the same for the toeplitz
tridi_matrix = TridiagonalToeplitz(3, a, b)
tridi_solution = [0, 0, 0, 0, 0, 0, 0, 0]
for j in range(N):
    vector[j] = 1
    tridi_solution[j] = HHL().solve(matrix, vector)
    vector[j] = 0


# In[28]:


from qiskit.quantum_info import Statevector


for k in range(N):
    w = np.real(Statevector(naive_hhl_solution[k].state).data)
    w = w[0:8]
    u = np.real(Statevector(tridi_solution[k].state).data)
    print(u)
    print(w)


# In[ ]:


naive_full_vector = np.real(naive_sv[0:8])
tridi_full_vector = np.real(tridi_sv)
print(naive_full_vector)
print(tridi_full_vector)
print(vector)
# lets define relative error based on the difference in distance and not in norm 
def dist(v1,v2):
    dist = 0
    for i in range(len(vector)):
        delta = (v1[i]-v2[i])**2
        distance = dist + delta
        print(delta)
    delta = np.sqrt(delta)
    return distance
r1 = dist(naive_full_vector,vector)
r2 = dist(tridi_full_vector,vector)
print([r1, r2])


# In[ ]:


from scipy.sparse import diags

num_qubits = 2
matrix_size = 2 ** num_qubits
# entries of the tridiagonal Toeplitz symmetric matrix
a = 1
b = -1/3

matrix = diags([b, a, b], [-1, 0, 1], shape=(matrix_size, matrix_size)).toarray()
vector = np.array([1] + [0]*(matrix_size - 1))

# run the algorithms
classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
naive_hhl_solution = HHL().solve(matrix, vector)
tridi_matrix = TridiagonalToeplitz(num_qubits, a, b)
tridi_solution = HHL().solve(tridi_matrix, vector)

print('classical euclidean norm:', classical_solution.euclidean_norm)
print('naive euclidean norm:', naive_hhl_solution.euclidean_norm)
print('tridiagonal euclidean norm:', tridi_solution.euclidean_norm)


# In[ ]:


print('naive state:')
print(naive_hhl_solution.state)
print('tridiagonal state:')
print(tridi_solution.state)


# In[ ]:


from qiskit import transpile

naive_qc = transpile(naive_hhl_solution.state,basis_gates=['id', 'rz', 'sx', 'x', 'cx'])

