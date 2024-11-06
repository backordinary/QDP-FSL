# https://github.com/klumel8/VQE/blob/b38c5b8625017c5c967ad262c7389fd184926a0c/Sasha/MeasureHamiltonian.py
import numpy as np
import math

from qiskit import(
	QuantumCircuit,
	execute,
	Aer,
	QuantumRegister,
	ClassicalRegister,
	IBMQ
	)
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plot_histogram

loadString = "\n========================================\n"
print(loadString+"Libraries imported. Proceeding with loading account and loading simulator"+loadString)

""" Now we define the desired vector, but in reality we would like to have a parametrised variational form in the quantum system iself. """
desired_vector = [1/math.sqrt(2) * complex(1,0),
	0,
	0,
	1/math.sqrt(2) * complex(1,0)
	]

#IBMQ.save_account('c650f2a6e47504e201bd6e1422e89efbb3b1ac24264be980a0555746424cd5d8469e1511b86a10f0f3fd1581245d864bcbe03b0802ce48dff35adc57a44da5f6')
IBMQ.load_account()

backend = Aer.get_backend('qasm_simulator')

print(loadString+"Account and simulator are loaded. Proceeding with job"+loadString)

"""Initialise parameters"""

qNum = 2
shots = 1000

# H = h*sum(z_i)+Jx*sum(x_i*x_i+1)
h = 1
Jx = 1


qrD = QuantumRegister(qNum)
crD = ClassicalRegister(qNum)
qrI = QuantumRegister(3)
crI = ClassicalRegister(1)


""" The Z part of the Hamiltonian""" 
""" Define the circuit"""
circuitZ = QuantumCircuit(qrD, crD)

circuitZ.initialize(desired_vector, qrD)

circuitZ.measure(qrD,crD)

print(circuitZ.draw())

""" Execute the ciruit """
jobZ = execute(circuitZ,backend, shots=shots)
resultZ = jobZ.result()
countsZ = resultZ.get_counts()
keysZ = list(countsZ.keys())

print(countsZ)

""" Calculate the Expecation value for Z """
EZ = 0
for key in keysZ:

	EZ += key.count('0')*countsZ[key]*h
	EZ -= key.count('1')*countsZ[key]*h
EZ /= shots
print("Expected value for Z component of energy is: ",EZ)

""" The X part of the Hamiltonian """
""" Define the circuit """
circuitXX = QuantumCircuit(qrI, crI)

circuitXX.initialize(desired_vector, qrI[0:2])

circuitXX.h(qrI[2])
circuitXX.cx(qrI[2],qrI[0])
circuitXX.cx(qrI[2],qrI[1])
circuitXX.h(qrI[2])

circuitXX.measure(qrI[2],crI)

print(circuitXX.draw())

""" Execute the ciruit """
jobXX = execute(circuitXX,backend, shots=shots)
resultXX = jobXX.result()
countsXX = resultXX.get_counts()
keysXX = list(countsXX.keys())

print(countsXX)

""" Calculate the Expecation value for XX """
EXX = 0
for key in keysXX:
	EXX += key.count('0')*countsXX[key]*Jx
	EXX -= key.count('1')*countsXX[key]*Jx
	print(EXX)
EXX /= shots
print("Expected value for X component of energy is: ",EXX)

E = EZ+EXX
print("Expected value for energy is: ",E)

