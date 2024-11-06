# https://github.com/Naarsk/qiskit-qubic-provider/blob/80f6b92091cd4632e6c003507cb1e49a091bb595/attic/test2.py
#!/usr/bin/env python3
"""
Created on Wed Apr 21 12:05:36 2021

@author: Leonardo
"""
#Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qubic_provider import QUBICProvider
#from qubic_job import QUBICJob

#...!...!....................
def circA():
    qc = QuantumCircuit(2)   #create a new 2-qubit quantum circuit
    qc.h(0)                  #add an hadamard gate on the 1st qubit
    qc.cx(0,1)               #add a cnot gate on the 2 qubits
    qc.measure_all()
    return qc

#...!...!....................
def circB(): # a random non-trivial 2Q circuit
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.t(1)
    bell.delay(4) # unit is dt
    bell.cx(0,1)
    bell.delay(4) # unit is dt
    bell.x(0)
    bell.h(1)
    bell.measure_all()
    return bell


#=================================
#=================================
#  M A I N 
#=================================
#=================================

#set the provider
provider = QUBICProvider()  

#set the backend
backend = provider.backends.qubic_backend

#create the circuit
qc=circA()
print('qc0');print(qc)
#transpile

trans_qc = transpile(qc, backend, basis_gates=['p','sx','cx'], optimization_level=1)
print('qc1');print(trans_qc)

#directly run the circuit (aka print the FakePut.txt) without need to assemble
job = backend.run(trans_qc)

print('\n cat FakePut.txt\n')
