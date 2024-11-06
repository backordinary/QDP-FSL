# https://github.com/anartify/qcg_qsolve/blob/85d7a506c5922d080628f6ad84c454a3b8994249/hhl_algo.py
import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.quantum_info import Statevector 
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver

def check_correct_hermitian(matrix, vector) : 
    
    if (np.all(np.abs(matrix - matrix.conj().T)<1e-8)) :
        flag = 0
        return matrix, vector, flag
    
    else : 
        mat1 = np.column_stack((np.zeros((2, 2)), matrix.conj().T))
        mat2 = np.column_stack((matrix, np.zeros((2,2)))) 
        mat = np.row_stack((mat1, mat2))
        vec = np.row_stack((np.zeros((2,1)), vector.reshape(2,1)))
        flag = 1
        return mat, vec, flag

def hhl_solver(matrix, vector):
    
    mat, vec, flag = check_correct_hermitian(matrix, vector)
    
    if flag == 0 :
        
        sol = HHL().solve(mat, vec) 
        data = Statevector(sol.state).data 
        vec1 = np.array([data[8], data[9]])
        vec1 = np.real(vec1)
        vec1 = sol.euclidean_norm*vec1/np.linalg.norm(vec1)
        return sol.euclidean_norm, vec1
        
    elif flag == 1 : 
        
        sol = HHL().solve(mat, vec) 
        data = Statevector(sol.state).data 
        vec1 = np.array([data[32], data[33], data[34], data[35]])
        vec1 = np.real(vec1)
        vec1 = sol.euclidean_norm*vec1/np.linalg.norm(vec1) 
        vec1 = vec1[0:3]
        return sol.euclidean_norm, vec1

from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver

def check_classical(matrix, vector):
    
    classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))
    classical_vector = classical_solution.state
    classical_norm = classical_solution.euclidean_norm
    return classical_vector, classical_norm

def hhl(A,b):
    
    str1 = "HHL is not godly, noise, depth, igh, expecting great errors !!!" 
    str2 = "Expecting intermediate errors"
    str3 = "Worked like a charm"

    import re
    result = re.findall(r"[-+]?\d*\.\d+|\d+", A)
    matrix = np.zeros(4)
    matrix[0] = float(result[0])
    matrix[1] = float(result[1])
    matrix[2] = float(result[2])
    matrix[3] = float(result[3])

    matrix = matrix.reshape((2,2))

    result = re.findall(r"[-+]?\d*\.\d+|\d+", b)
    vector = np.zeros(2)
    vector[0] = float(result[0])
    vector[1] = float(result[1])

    
    quantum_norm, quantum_vector = hhl_solver(matrix,vector)
    classical_vector, classical_norm =check_classical(matrix,vector)
    
    quantum_vector = quantum_vector*np.linalg.norm(vector)
    classical_vector = classical_vector*np.linalg.norm(vector)

    if ( abs(quantum_norm - classical_norm) <1e-4) : 
        stmt = str(str3) + "... Classical_x = " + str(classical_vector) + "... Quantum_x = " +str(quantum_vector)
        return stmt 
    
    elif ( abs(quantum_norm - classical_norm) >1e-4  and abs(quantum_norm - classical_norm) <1e-1):
        stmt = str(str2) + "... Classical_x = " + str(classical_vector) + "... Quantum_x = " +str(quantum_vector)
        return stmt
    else:
        stmt = str(str1) + "... Classical_x = " + str(classical_vector)
        return stmt