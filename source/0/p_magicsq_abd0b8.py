# https://github.com/Apress/practical-quantum-computing/blob/227861a105f641e733fc568468e417aba7f098c1/Ch07/p_magicsq.py
# https://github.com/QISKit/qiskit-tutorial/blob/old_tutorial_format/5_games/quantum_magic_square.ipynb
# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
	raise Exception('Please use Python version 3.5 or greater.')

# useful additional packages 
import matplotlib.pyplot as plt
import numpy as np

# useful math functions
from math import pi, cos, acos, sqrt
import random

# importing the QISKit
# Q Experience config
sys.path.append('../Config/')
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

# Prior to the start of the game, Alice and Bob share the following quantum state
Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# 4 qubits (Alice = 2, Bob = 2)
N = 4

# Creating registers
qr = Q_program.create_quantum_register("qr", N)

# for recording the measurement on qr
cr = Q_program.create_classical_register("cr", N)

circuitName = "sharedEntangled"
sharedEntangled = Q_program.create_circuit(circuitName, [qr], [cr])
sharedEntangled.h(qr[0])
sharedEntangled.h(qr[1])
sharedEntangled.z(qr[0])
sharedEntangled.z(qr[1])
sharedEntangled.cx(qr[0], qr[0+2])
sharedEntangled.cx(qr[1], qr[1+2])
sharedEntangled.x(qr[2])
sharedEntangled.x(qr[3])
#------  circuits of Alice's and Bob's operations.
#we first define controlled-u gates required to assign phases 
from math import pi

def ch(qProg, a, b):
    """ Controlled-Hadamard gate """
    qProg.h(b)
    qProg.sdg(b)
    qProg.cx(a, b)
    qProg.h(b)
    qProg.t(b)
    qProg.cx(a, b)
    qProg.t(b)
    qProg.h(b)
    qProg.s(b)
    qProg.x(b)
    qProg.s(a)
    return qProg

def cu1pi2(qProg, c, t):
    """ Controlled-u1(phi/2) gate """
    qProg.u1(pi/4.0, c)
    qProg.cx(c, t)
    qProg.u1(-pi/4.0, t)
    qProg.cx(c, t)
    qProg.u1(pi/4.0, t)
    return qProg

def cu3pi2(qProg, c, t):
    """ Controlled-u3(pi/2, -pi/2, pi/2) gate """
    qProg.u1(pi/2.0, t)
    qProg.cx(c, t)
    qProg.u3(-pi/4.0, 0, 0, t)
    qProg.cx(c, t)
    qProg.u3(pi/4.0, -pi/2.0, 0, t)
    return qProg

#--------------------------------------------------------------------------
# Define circuits used by Alice and Bob for each of their inputs: 1,2,3
# dictionary for Alice's operations/circuits
aliceCircuits = {}
circuitName = "Alice"+str(1)
aliceCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
theCircuit = aliceCircuits[circuitName]

if 1 == 1:
    #the circuit of A_1
    theCircuit.x(qr[1])
    theCircuit.cx(qr[1], qr[0])
    theCircuit = cu1pi2(theCircuit, qr[1], qr[0])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit = cu3pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit = ch(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit.cx(qr[1], qr[0])
    theCircuit.x(qr[1])
    
elif 1 == 2:
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.h(qr[0])
    theCircuit.h(qr[1])

elif 1 == 3:
    theCircuit.cz(qr[0], qr[1])
    theCircuit.swap(qr[0], qr[1]) # not supported in composer
    theCircuit.h(qr[0])
    theCircuit.h(qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit.cz(qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    
#measure the first two qubits in the computational basis
theCircuit.measure(qr[0], cr[0])
theCircuit.measure(qr[1], cr[1])
circuitName = "Alice"+str(2)
aliceCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
theCircuit = aliceCircuits[circuitName]

if 2 == 1:
    #the circuit of A_1
    theCircuit.x(qr[1])
    theCircuit.cx(qr[1], qr[0])
    theCircuit = cu1pi2(theCircuit, qr[1], qr[0])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit = cu3pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit = ch(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit.cx(qr[1], qr[0])
    theCircuit.x(qr[1])
    
elif 2 == 2:
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.h(qr[0])
    theCircuit.h(qr[1])

elif 2 == 3:
    theCircuit.cz(qr[0], qr[1])
    theCircuit.swap(qr[0], qr[1]) # not supported in composer
    theCircuit.h(qr[0])
    theCircuit.h(qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit.cz(qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    
#measure the first two qubits in the computational basis
theCircuit.measure(qr[0], cr[0])
theCircuit.measure(qr[1], cr[1])
circuitName = "Alice"+str(3)
aliceCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
theCircuit = aliceCircuits[circuitName]

if 3 == 1:
    #the circuit of A_1
    theCircuit.x(qr[1])
    theCircuit.cx(qr[1], qr[0])
    theCircuit = cu1pi2(theCircuit, qr[1], qr[0])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit = cu3pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit = ch(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit.cx(qr[1], qr[0])
    theCircuit.x(qr[1])
    
elif 3 == 2:
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit = cu1pi2(theCircuit, qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.h(qr[0])
    theCircuit.h(qr[1])

elif 3 == 3:
    theCircuit.cz(qr[0], qr[1])
    theCircuit.swap(qr[0], qr[1]) # not supported in composer
    theCircuit.h(qr[0])
    theCircuit.h(qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    theCircuit.cz(qr[0], qr[1])
    theCircuit.x(qr[0])
    theCircuit.x(qr[1])
    
#measure the first two qubits in the computational basis
theCircuit.measure(qr[0], cr[0])
theCircuit.measure(qr[1], cr[1])

# dictionary for Bob's operations/circuits
bobCircuits = {}
circuitName = "Bob"+str(1)
bobCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
theCircuit = bobCircuits[circuitName]
if 1 == 1:
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[3])
    theCircuit.u1(pi/2.0, qr[2])
    theCircuit.x(qr[2])
    theCircuit.z(qr[2])
    theCircuit.cx(qr[2], qr[3])
    theCircuit.cx(qr[3], qr[2])
    theCircuit.h(qr[2])
    theCircuit.h(qr[3])
    theCircuit.x(qr[3])
    theCircuit = cu1pi2(theCircuit, qr[2], qr[3])
    theCircuit.x(qr[2])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    
elif 1 == 2:
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[3])
    theCircuit.u1(pi/2.0, qr[3])
    theCircuit.cx(qr[2], qr[3])
    theCircuit.h(qr[2])
    theCircuit.h(qr[3])

elif 1 == 3:
    theCircuit.cx(qr[3], qr[2])
    theCircuit.x(qr[3])
    theCircuit.h(qr[3])
    
#measure the third and fourth qubits in the computational basis
theCircuit.measure(qr[2], cr[2])
theCircuit.measure(qr[3], cr[3])
circuitName = "Bob"+str(2)
bobCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
theCircuit = bobCircuits[circuitName]
if 2 == 1:
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[3])
    theCircuit.u1(pi/2.0, qr[2])
    theCircuit.x(qr[2])
    theCircuit.z(qr[2])
    theCircuit.cx(qr[2], qr[3])
    theCircuit.cx(qr[3], qr[2])
    theCircuit.h(qr[2])
    theCircuit.h(qr[3])
    theCircuit.x(qr[3])
    theCircuit = cu1pi2(theCircuit, qr[2], qr[3])
    theCircuit.x(qr[2])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    
elif 2 == 2:
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[3])
    theCircuit.u1(pi/2.0, qr[3])
    theCircuit.cx(qr[2], qr[3])
    theCircuit.h(qr[2])
    theCircuit.h(qr[3])

elif 2 == 3:
    theCircuit.cx(qr[3], qr[2])
    theCircuit.x(qr[3])
    theCircuit.h(qr[3])
    
#measure the third and fourth qubits in the computational basis
theCircuit.measure(qr[2], cr[2])
theCircuit.measure(qr[3], cr[3])
circuitName = "Bob"+str(3)
bobCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
theCircuit = bobCircuits[circuitName]
if 3 == 1:
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[3])
    theCircuit.u1(pi/2.0, qr[2])
    theCircuit.x(qr[2])
    theCircuit.z(qr[2])
    theCircuit.cx(qr[2], qr[3])
    theCircuit.cx(qr[3], qr[2])
    theCircuit.h(qr[2])
    theCircuit.h(qr[3])
    theCircuit.x(qr[3])
    theCircuit = cu1pi2(theCircuit, qr[2], qr[3])
    theCircuit.x(qr[2])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    
elif 3 == 2:
    theCircuit.x(qr[2])
    theCircuit.x(qr[3])
    theCircuit.cz(qr[2], qr[3])
    theCircuit.x(qr[3])
    theCircuit.u1(pi/2.0, qr[3])
    theCircuit.cx(qr[2], qr[3])
    theCircuit.h(qr[2])
    theCircuit.h(qr[3])

elif 3 == 3:
    theCircuit.cx(qr[3], qr[2])
    theCircuit.x(qr[3])
    theCircuit.h(qr[3])
    
#measure the third and fourth qubits in the computational basis
theCircuit.measure(qr[2], cr[2])
theCircuit.measure(qr[3], cr[3])

#################################################
# A quantum program for one round of the game
# backend: device name or simulator
# real_dev: Tru to run in a real device
# shots: number of shots
#################################################
def one_round(backend, real_dev, shots = 1):
	#generate random integers
	a, b = random.randint(1,3), random.randint(1,3) 
	#a, b = 3, random.randint(1,3) 
	print("Magic Square  a = " + str(a) + " b = " + str(b) + " Device: " + backend)

	aliceCircuit 	= aliceCircuits["Alice" + str(a)]
	bobCircuit 		= bobCircuits["Bob" + str(b)]
	circuitName 	= "Alice" + str(a) + "Bob" + str(b)
	
	Q_program.add_circuit(circuitName, sharedEntangled + aliceCircuit + bobCircuit)

	print(Q_program.get_qasm(circuitName))

	if real_dev:
		shots			= 10
		device_cfg 		= Q_program.get_backend_configuration(backend)
		device_coupling = device_cfg['coupling_map']
		
		print("Running in real device " + backend + " Couplings: " + str(device_coupling))
		results 		= Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=device_coupling)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer 	= results.get_counts(circuitName)
	
	print("Device = " + backend + " counts: " + str(answer))

	for key in answer.keys():
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:#the sume of Alice answer must be even
			aliceAnswer.append(0) 
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:#the sum of Bob answer must be odd
			bobAnswer.append(0)   
		else:
			bobAnswer.append(1)
		break

	print("Alice answer for a = ", a, "is", aliceAnswer)
	print("Bob answer for b = ", b, "is", bobAnswer)

	if(aliceAnswer[b-1] != bobAnswer[a-1]): #check if the intersection of their answers is the same
		print("Alice and Bob lost")
	else:
		print("Alice and Bob won")

#################################################
# A quantum program for all rounds
# backend: device name or simulator
# real_dev: Tru to run in a real device
# shots: # of shots
#################################################
def all_rounds(backend, real_dev, shots=10):
	nWins = 0
	nLost = 0
	print("For a = " + str(1) + " , b = " + str(1))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(1)]
	bobCircuit 		= bobCircuits["Bob" + str(1)]
	circuitName 	= "Alice" + str(1) + "Bob"+str(1)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[1-1] != bobAnswer[1-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(1) + " , b = " + str(2))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(1)]
	bobCircuit 		= bobCircuits["Bob" + str(2)]
	circuitName 	= "Alice" + str(1) + "Bob"+str(2)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[2-1] != bobAnswer[1-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(1) + " , b = " + str(3))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(1)]
	bobCircuit 		= bobCircuits["Bob" + str(3)]
	circuitName 	= "Alice" + str(1) + "Bob"+str(3)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[3-1] != bobAnswer[1-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(2) + " , b = " + str(1))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(2)]
	bobCircuit 		= bobCircuits["Bob" + str(1)]
	circuitName 	= "Alice" + str(2) + "Bob"+str(1)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[1-1] != bobAnswer[2-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(2) + " , b = " + str(2))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(2)]
	bobCircuit 		= bobCircuits["Bob" + str(2)]
	circuitName 	= "Alice" + str(2) + "Bob"+str(2)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[2-1] != bobAnswer[2-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(2) + " , b = " + str(3))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(2)]
	bobCircuit 		= bobCircuits["Bob" + str(3)]
	circuitName 	= "Alice" + str(2) + "Bob"+str(3)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[3-1] != bobAnswer[2-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(3) + " , b = " + str(1))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(3)]
	bobCircuit 		= bobCircuits["Bob" + str(1)]
	circuitName 	= "Alice" + str(3) + "Bob"+str(1)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[1-1] != bobAnswer[3-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(3) + " , b = " + str(2))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(3)]
	bobCircuit 		= bobCircuits["Bob" + str(2)]
	circuitName 	= "Alice" + str(3) + "Bob"+str(2)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[2-1] != bobAnswer[3-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")
	print("For a = " + str(3) + " , b = " + str(3))
	rWins = 0
	rLost = 0
	
	aliceCircuit 	= aliceCircuits["Alice" + str(3)]
	bobCircuit 		= bobCircuits["Bob" + str(3)]
	circuitName 	= "Alice" + str(3) + "Bob"+str(3)
	Q_program.add_circuit(circuitName, sharedEntangled+aliceCircuit+bobCircuit)

	if real_dev:
		ibmqx2_backend = Q_program.get_backend_configuration(backend)
		ibmqx2_coupling = ibmqx2_backend['coupling_map']
		results = Q_program.execute([circuitName], backend=backend, shots=shots, coupling_map=ibmqx2_coupling, max_credits=3, timeout=240)
	else:
		results = Q_program.execute([circuitName], backend=backend, shots=shots)
		
	answer = results.get_counts(circuitName)

	for key in answer.keys():
		kfreq = answer[key] #frequencies of keys obtained from measurements
		aliceAnswer = [int(key[-1]), int(key[-2])]
		bobAnswer   = [int(key[-3]), int(key[-4])]
		if sum(aliceAnswer) % 2 == 0:
			aliceAnswer.append(0)
		else:
			aliceAnswer.append(1)
		if sum(bobAnswer) % 2 == 1:
			bobAnswer.append(0)
		else:
			bobAnswer.append(1)

		print(backend + " answer: " + key + " Alice: " + str(aliceAnswer) + " Bob:" + str(bobAnswer))
		#print("Alice answer for a = ", a, "is", aliceAnswer)
		#print("Bob answer for b = ", b, "is", bobAnswer)
			
		if(aliceAnswer[3-1] != bobAnswer[3-1]):
			#print(a, b, "Alice and Bob lost")
			nLost += kfreq
			rLost += kfreq
		else:
			#print(a, b, "Alice and Bob won")
			nWins += kfreq
			rWins += kfreq
	print("\t#wins = ", rWins, "out of ", shots, "shots")

	print("Number of Games = ", nWins+nLost)
	print("Number of Wins = ", nWins)
	print("Winning probabilities = ", (nWins*100.0)/(nWins+nLost))

	
#################################################
# main
#################################################
if __name__ ==  '__main__':
	#backend = "local_qasm_simulator"
	backend = "ibmqx_qasm_simulator"
	#backend = "ibmqx2"
	real_dev = False

	#one_round(backend, real_dev)
	all_rounds(backend, real_dev)
	