# https://github.com/StreakSharn/Grovers-Algorithm-Visualisation/blob/226b61049a7911fa26405fe40f97170ffc9d50ab/src/grover_algorithm.py
# IMPORTS
import qiskit
import numpy as np
from progressbar import progressbar
from random import randint


############## FUNCTIONS ##################

def Grover(n,hidden_bits,iterations):
    circuit = qiskit.QuantumCircuit(n+1,n)    # one extra qubit for the Auxiliary, and a classical bit for each Query qubit

    for i in range(n):
        circuit.h(i)    # place Query into equal superposition

    circuit.x(n)
    circuit.h(n)  # bring Auxiliary into '1', then '-' state for Phase Kickback

    circuit.barrier()

    for _i in progressbar(range(iterations)): # amplitude amplification
        circuit = Oracle(circuit,n,hidden_bits)
        circuit = Diffuser(circuit,n) #Amplitude Amplification

    return circuit

def Oracle(circuit,n,hidden_bits):

    circuit.barrier()

    for i in range(len(hidden_bits)):
        if hidden_bits[i] == '0':
            circuit.x(i)    #apply X Gates to any '0's in the hidden bit string so they can activate MCX

    circuit.mcx(list(range(n)),n) # check if qubit is perfectly correct

    for i in range(len(hidden_bits)):
        if hidden_bits[i] == '0':
            circuit.x(i)   #finish wrapping X Gates around '0's to hide any

    circuit.barrier()

    return circuit

def Diffuser(circuit,n):

    for i in range(n):
        circuit.h(i)  #take 's' out of superposition
        circuit.x(i)  #flip to |11..> to activate MCX

    circuit.mcx(list(range(n)),n) # apply Phase Kickback

    for i in range(n):
        circuit.x(i)
        circuit.h(i) #undo X Gate and place back into superposition

    circuit.barrier()

    return circuit



################ MAIN ###################

def Get_Grover_Statevectors():
    '''
    Returns a list of Grover's Algorithm's Statevectors at various points in the Algorithm.
    '''

    # parameters
    n = 7   # size of Query and hidden bit string
    hidden_bits = randint(0,2**n-1)     #random n-digit number
    hidden_bits = format(hidden_bits,'0'+str(n)+'b')    #convert number to binary string


    max_iterations = int(np.floor((np.pi/4)*np.sqrt(2**n))) #formula for number of iterations

    statevectors = []

    for iterations in progressbar(range(max_iterations+1),redirect_stdout=True): # loops through circuit progress

        circuit = Grover(n,hidden_bits,iterations)

        simulator = qiskit.Aer.get_backend('statevector_simulator')
        job = qiskit.execute(circuit,simulator,shots=1)
        statevec = job.result().get_statevector(circuit)    # simulation

        statevectors.append(statevec[len(statevec)//2:]) # exclude auxiliary/output qubit from statevector

    return statevectors
