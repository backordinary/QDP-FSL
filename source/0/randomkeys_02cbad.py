# https://github.com/TheSleepyKing/UWCProject/blob/acdec6c2af88edb720999e55561646a078fea4b3/randomKeys.py
# Useful packages
from qiskit import *
import math
import random


#import qiskit
from qiskit import BasicAer, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

def Quantum_key(length_str, H_gate):
    #output variables used to access quantum computer results atthe end of th function
    q_output = ''
    
    #starting up quantum circuit information
    backend = BasicAer.get_backend('qasm_simulator')
    circuits =['circuit']
    
    #run citcuit in batches of 10 qubits for fastest result.
    #The result from each run will be appended and then clipped down to the right n size
    n = length_str
    output = ''
    for i in range(math.ceil(n/16)):
        #intialize quantum registers for circuit
        q = QuantumRegister(16, 'q')
        c = ClassicalRegister(16, 'c')

        circuit = QuantumCircuit(q,c)
        
        #create temp_n number of qubits all in superposition 
        for i in range(16):
            for j in range(0,H_gate):
                circuit.h(q[i]) #the .h gate is the Hadamard gate that makes superposition
            circuit.measure(q[i],c[i])
            
        #execute circuit and extract 0s and 1s from key
        sim = Aer.get_backend('aer_simulator') 
        # results = sim.run(circuit).result()
        # count = results.get_counts()
        result = execute(circuit,backend,shots=1).result()
        result_key = list(result.get_counts(circuit).keys())
        output = result_key[0]
        q_output += output
    #return output clipped to size of desired string length
    return q_output[:n]  



def psuedo_key(string_length):
    #output variables used to access quantum computer results atthe end of th function
    output = ''
    
    n = string_length
    for i in range(n):
        temp_n = str(random.randint(0,1))
        output += temp_n
    #return output clipped to size of desired string length
    return output[:n] 


def text_conversion(message):

    file = open(message)
    string = file.read()
    return string
