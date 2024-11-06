# https://github.com/BenWhiteside/Quantum/blob/9ae1de847d4632279222c091d69a0b570ab43333/testBackends.py
from qiskit import Aer
from qiskit import IBMQ

for backend in Aer.backends():
    print(backend.name())


for backend in IBMQ.providers():
    print(backend.name())