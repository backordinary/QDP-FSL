# https://github.com/urwin419/QiskitChecker/blob/fbd0aa619dffdca4cca1281e92640bb6278c0c22/SimonAlgorithm/Remove_Mutant_2.py
import numpy as np
import qiskit
#Code based on Qiskit offcial Textbook
#Create the oracle for the secret string
def oracle(circuit: qiskit.QuantumCircuit, bitstring: str):
    
    #Get the length of the bitstring 
    length = len(bitstring)

    #Copy every qubit from the first register to the second register
    for i in range(length):
        circuit.cx(i, length+i)

    #Create a 1-to-1 or 2-to-1 mapping:
    #Get the index of the last 1 in bitstring
    l = bitstring.rfind('1')

    #Flip the qubit with index in the second register for bitstring[index] is 1
    for index, s in enumerate(bitstring):
        if s == '1' and l != -1:
            circuit.cx(l, length+index)

    #Create a random permutation
    #Get random permutation of n qubits
    permutation = list(np.random.permutation(length))

    #Record initial positions
    ini_p = list(range(length))
    for i in range(length-1):
        if ini_p[i] != permutation[i]:
            j = permutation.index(ini_p[i])
            #Swap qubits
            circuit.swap(length+i, length+j)
            #Record the positions of swapped qubits
            ini_p[i], ini_p[j] = ini_p[j], ini_p[i]
                
    #Flip the qubits randomly
    for i in range(length):
        if np.random.random() > 0.5:
            circuit.x(length+i)

    return circuit

def algorithm(bitstring: str):

    #Get the length of the bitstring
    length = len(bitstring)

    #Create the quantum circuits
    simon_circuit = qiskit.QuantumCircuit(length*2, length)

    #Apply H-gates to all qubits
    simon_circuit.h(range(length))

    #Apply the oracle created
    simon_circuit = oracle(simon_circuit,bitstring)

    #Apply H-gates to all qubits
    """REMOVE MUTANT"""
    #simon_circuit.h(range(length))
    """REMOVE MUTANT"""

    #Measure all the qubits
    simon_circuit.measure(range(length), range(length))

    return simon_circuit


#Calculate the dot-product
def dotp(b, z):
    
    #Calculate the dot-product by mutiply each position and add them together
    accum = 0
    for i  in range(len(b)):
        accum += int(b[i]) * int(z[i])
    
    #Return the result by modulo 2
    return accum % 2

#Run the simulation with the given circuit
def simon(bitstring:str):
    
    #Get the simulator
    qasm_simulator = qiskit.Aer.get_backend('qasm_simulator')

    #Execute the Simon's algorithm
    job = qiskit.execute(algorithm(bitstring),qasm_simulator, shots=10000)

    # Get results
    result = job.result().get_counts()
    
    return result