# https://github.com/lionelchg/LesAtomesDeSavoie/blob/9e0c37ce2ea634446b0406ceea90d9432461ae86/BIG/problem_toquad.py
import numpy as np

from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer
from qiskit.optimization.problems import QuadraticProgram
from qiskit import BasicAer
from docplex.mp.model import Model

def Qmatrix(filter_len, chain_len):
    seek_len = chain_len * filter_len #taille de Z
    qMatrix = np.full(shape=(filter_len * chain_len, filter_len * chain_len), fill_value=0)
    matrix_list = []
    for k in range(1, max(chain_len, filter_len)):
        retard_matrix = np.full(shape=(filter_len, chain_len), fill_value=0)
        for i in range(filter_len):
            for j in range(chain_len):
                if(i<j and i+chain_len-k<chain_len):
                    retard_matrix[i,i+chain_len-k] = 1
        matrix_list.append(retard_matrix)
        print(retard_matrix)

    for matrix in matrix_list:
        flatten_q = np.reshape(matrix, (seek_len, 1))
        transposed_q = flatten_q.transpose()
        quadratic_term = np.dot(flatten_q, transposed_q)

        qMatrix = qMatrix + quadratic_term
    linear_param = np.zeros((chain_len)*(filter_len))
    print("Matrice R:\n", qMatrix, qMatrix.shape)

    return qMatrix

if __name__ == '__main__':
    chain_len = 5 #taille de X
    qmat = Qmatrix(5, 5)

    model = Model()
    x0 = model.binary_var(name='bit0')
    x1 = model.binary_var(name='bit1')
    x2 = model.binary_var(name='bit2')
    x3 = model.binary_var(name='bit3')
    x4 = model.binary_var(name='bit4')

