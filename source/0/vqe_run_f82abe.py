# https://github.com/DanorRon/qiskit-vqe/blob/c71838739c00b4947c8ce50cd0661911211bbada/1d%20TFIM/Real%20Quantum%20Computer/vqe_run.py
import qiskit
from qiskit import Aer, IBMQ, QuantumCircuit
import vqe_functions
from vqe_functions import VQE

backend = Aer.get_backend('aer_simulator')

#IBMQ.save_account('fb3e08528d7c6638591c88e8a957996475aea67811f7dee0916862a946430748a2791e32a01f89e7e3d7fe7030855562e3bafa9014591b9ea50eeadb220858f3')
#provider = IBMQ.load_account()
#print(IBMQ.providers())
#provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
#print(provider.backends())
#backend = provider.get_backend('ibmq_qasm_simulator')


num_qubits = 3
layers = 5
shots = 10000
h_zz = -1
h_z = 1
h_x = 1
maxiter = 100

vqe = VQE(num_qubits, layers, shots, h_zz, h_z, h_x, maxiter, backend)
initial_thetas = vqe.initial_params()
solution = vqe.optimize(vqe.loss, initial_thetas)
print('Final thetas: ' + str(solution.x))
print('Final expected value (ground state energy): ' + str(solution.fun))