# https://github.com/JoelMathewC/Quantum-Computation-Project/blob/1d91f12f883bb3e2b8a540de26a9ab18e0152c93/Quantum%20Approach/qiskit-backends.py
from qiskit import IBMQ, Aer

provider = IBMQ.load_account()
available_cloud_backends = provider.backends() 
print('\nHere is the list of cloud backends that are available to you:')
for i in available_cloud_backends: print(i)

available_local_backends = Aer.backends() 
print('\nHere is the list of local backends that are available to you: ')
for i in available_local_backends: print(i)