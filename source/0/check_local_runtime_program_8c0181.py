# https://github.com/paulmatt67/quantum_packer_public/blob/b37d57d4a1682579dd4025115215d0ef0f2eb8bc/check_local_runtime_program.py
from qiskit.providers.ibmq.runtime import UserMessenger
from qiskit import BasicAer
from qiskit import Aer
from TSP_RUNTIME_SOLVER import *
from qiskit_methods.config import API_TOKEN_DE, API_TOKEN_US, API_URL_DE
from qiskit_methods.backends import show_num_qubits_for_provider
from qiskit_methods.backends import find_least_busy_backend
from qiskit import IBMQ


# 1. Select a local simulator
#backend = BasicAer.get_backend('qasm_simulator')
# 2. Select a local GPU simulator
backend = Aer.get_backend('aer_simulator')
#backend.set_options(device='GPU')

# Create a user messenger
user_messenger = UserMessenger()

# Create a dictionary of TSP problem instances
problem_instances = {}
D1 = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
D2 = [[0, 1, 2, 1], [1, 0, 3, 1], [2, 3, 0, 0], [1, 1, 0, 0]]
problem_instances.update({'TSP1 3x3': D1})
problem_instances.update({'TSP2 4x4': D2})

# LOCAL TEST OF THE QUBO RUNTIME SOLVER
solutions = main(backend, user_messenger, problem_instances)
