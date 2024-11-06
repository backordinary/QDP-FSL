# https://github.com/MADjnr/Quantum_projectsQ-/blob/e8e5d723250225073dcb1e94757a9e4127d9135d/Classicalcircuitswithquantumcomputation.py
from qiskit import Aer
for backend in Aer.backends():
    print(backend.name())
