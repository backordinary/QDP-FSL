# https://github.com/CFP106022272/HHL-algorithm/blob/1ddd621c21f75149dc87c0a3237cdcdd1aa3e2b4/HHL%20algorithm.py
#引入qiskit
import qiskit
import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.algorithms.linear_solvers.hhl import HHL

#建立樣本(矩陣、向量)
matrix = np.array([[1, -1/2], [-1/2, 1]])
vector = np.array([1, 0])

#用數學模組解樣本
classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
naive_hhl_solution = HHL().solve(matrix, vector)

#打印狀態
print('classical state:', classical_solution.state)
#打印狀態(以量子電路圖呈現其狀態)
print('naive state:') 
print(naive_hhl_solution.state)

#打印歐基里德範數(主要是為了之後量子狀態的重整)
print('classical Euclidean norm:', classical_solution.euclidean_norm)
print('naive Euclidean norm:', naive_hhl_solution.euclidean_norm)

##狀態重整化##
from qiskit.quantum_info import Statevector

naive_sv = Statevector(naive_hhl_solution.state).data

#我們實際有興趣的qubit序為 1000 (q0_0, q1_0(工作位元1_work qubit1), q1_1(工作位元2_work qubit2), q2_0(輔助位元_auxiliary qubit)) 對應十進位 8 以及 1001 對應十進位 9
naive_full_vector = np.array([naive_sv[8], naive_sv[9]])

#打印出目前向量
print('naive raw solution vector:', naive_full_vector)

#虛數部份非常小，推測其產生的原因來自於經典電腦的準確性，因此這部份可以忽略
naive_full_vector = np.real(naive_full_vector)

#打印重整化狀態
print('classical state:', classical_solution.state)

#量子狀態的重整化需要除以各自的範數，然後再乘上整個系統歸一化過的歐基里德範數
print('full naive solution vector:', naive_hhl_solution.euclidean_norm*naive_full_vector/np.linalg.norm(naive_full_vector))

