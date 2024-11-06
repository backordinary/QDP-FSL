# https://github.com/neilgupte75/CS_810_Quantum-Computing_Neil/blob/67697b0288014c72ab65277a04881c48c7351a19/Final%20Project%20shors/shors%20trial.py
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:06:08 2020

@author: Neil Gupte
"""
from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor

IBMQ.enable_account( ' aditya replace your token here 52b916f6b6934d0ff3a5754d98217dc1862bcb88cc04acc77147936d166cb28f67d4f60d6fd492b7') # Enter your API token here
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator') # Specifies the quantum device

print('\n Shors Algorithm')
print('--------------------')
print('\nExecuting...\n')

factors = Shor(15) #Function to run Shor's algorithm where 21 is the integer to be factored

result_dict = factors.run(QuantumInstance(backend, shots=1, skip_qobj_validation=False))
result = result_dict['factors'] # Get factors from results

print(result)
print('\nPress any key to close')

circuit =Shor.construct_circuit() #check if we show the circuit by any means there is a circuit available on ibmq dashboard but it is too large to display see if there is a way to show that circuit
circuit.draw()
