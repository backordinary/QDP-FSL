# https://github.com/lucasarens/QOSF_Mentorship_Application/blob/88367c54a2396b1e6c9fe4be7955e6605096ef48/Task1_Bonus_Solution1.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np, math
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram


#Input is the vector of integers
#Output state vector of the form ket{psi}
def initial_state(in_vector):
    
    #Check and get n
    n = len(in_vector)
    if n%2 != 0:
        print('Invalid Input - Vector must have size 2^n, with n>=2')
        return 0,0,0
    else: n = int(math.log(len(in_vector),2))
        
    #Find m
    m = len(np.binary_repr(max(in_vector)))
    
    #Binary vector with binary indices
    bi_vector_bi_index = [0]* 2**n
    
    #This vector will contain the integer values of the indices 
    #from the bi_vector_bi_index
    index_vector = [0]* 2**n

    for i in range(len(in_vector)):
        bi_vector_bi_index[i] = (str(np.binary_repr(in_vector[i],m)) + str(np.binary_repr(i,n)))
        index_vector[i] = int(bi_vector_bi_index[i],2)
 
    initial_state_vector = [0] * 2**(n+m)
    
    #Fill the initial_state_vector with amplitudes that respect
    #the normalization condition for each of the states of ket{psi}
    for i in index_vector:
        initial_state_vector[i] = 1/math.sqrt(len(index_vector))

    return initial_state_vector, n, m


###################################################

def oracle_qram(n,m):
    #same size
    qc = QuantumCircuit(m+n+n+1,n)
    #oracle for 0101...
    if 0 == 0:
        for i in range(n, m+n, 2):
            qc.x(i)
    else:
        #oracle for 1010...
        for i in range(n+1, m+n, 2):
            qc.x(i)
        
        #This for loop runs twice for each one of the solutions
    for j in range(2**(n)):
        
        #For each one of the binary adds we built its MCX gate
        #using the appropriate oracle
        bi_add = np.binary_repr(j,n)
        #To list of integers
        l = [int(i) for i in list(bi_add)]
         
        #index of ones
        index_ones = [i for i, z in enumerate(l) if z == 1]
        
        #Index of ones for Grover (3rd register)
        index_ones_grover = [i+m+n for i in index_ones]
        
    
        if len(index_ones) != 0:
            #Nots for 1st and 3rd register
            qc.x(index_ones)
            qc.x(index_ones_grover)
            
            
        #Mult control not gate - target is the last qubit of the 3rd register
        qc.mcx(list(range(0,m+n+n)), m+n+n, mode='noancilla')
            
        #Uncompute
        if len(index_ones) != 0:
            qc.x(index_ones)
            qc.x(index_ones_grover)
        
        
        #Uncompute Oracle - just after the first 2^n indices
    if 0 == 0:
        for i in range(n, m+n, 2):
            qc.x(i)
    else:
        for i in range(n+1, m+n, 2):
            qc.x(i)
             
    #oracle for 0101...
    if 1 == 0:
        for i in range(n, m+n, 2):
            qc.x(i)
    else:
        #oracle for 1010...
        for i in range(n+1, m+n, 2):
            qc.x(i)
        
        #This for loop runs twice for each one of the solutions
    for j in range(2**(n)):
        
        #For each one of the binary adds we built its MCX gate
        #using the appropriate oracle
        bi_add = np.binary_repr(j,n)
        #To list of integers
        l = [int(i) for i in list(bi_add)]
         
        #index of ones
        index_ones = [i for i, z in enumerate(l) if z == 1]
        
        #Index of ones for Grover (3rd register)
        index_ones_grover = [i+m+n for i in index_ones]
        
    
        if len(index_ones) != 0:
            #Nots for 1st and 3rd register
            qc.x(index_ones)
            qc.x(index_ones_grover)
            
            
        #Mult control not gate - target is the last qubit of the 3rd register
        qc.mcx(list(range(0,m+n+n)), m+n+n, mode='noancilla')
            
        #Uncompute
        if len(index_ones) != 0:
            qc.x(index_ones)
            qc.x(index_ones_grover)
        
        
        #Uncompute Oracle - just after the first 2^n indices
    if 1 == 0:
        for i in range(n, m+n, 2):
            qc.x(i)
    else:
        for i in range(n+1, m+n, 2):
            qc.x(i)
             
    return qc        


def grover_diffuser(n,m):
    #same size
    qc = QuantumCircuit(m+n+n+1,n)
    
    #first qubit of the 3rd register to start the diffuser
    #up until the last but one qubit
    for i in range(m+n,m+n+n):
        qc.h(i)
        qc.x(i)
    
    #Multi-controlled Z gate equals H's Multi-controlled X gate H's
    qc.h(m+n+n-1)
    qc.mct(list(range(m+n,m+n+n-1)), m+n+n-1)
    qc.h(m+n+n-1)
    
    
    for i in range(m+n,m+n+n):
        qc.x(i)
        qc.h(i)
        
    return qc


def sol1_bonus(input_vector):
    
    in_vector = list(input_vector)
    
    initial_state_vector, n, m = initial_state(in_vector)
    if n == 0:
        return 
    
    circuit = QuantumCircuit(m+n+n+1,n)
    circuit.initialize(initial_state_vector, list(range(n+m)))
    
    #3rd Register Setup |++...+->
    #Hadamard for Grover
    circuit.h(list(range(m+n,m+n+n)))
    
    circuit.x(m+n+n)
    circuit.h(m+n+n)

    oracle_qram_circuit = oracle_qram(n,m)
    
    circuit.compose(oracle_qram_circuit, inplace=True)
    
    diffuser = grover_diffuser(n,m)
   
    circuit.compose(diffuser, inplace=True)

    #Measuring
    for i in range(n):
        circuit.measure(m+n+i,i)

    backend = Aer.get_backend('statevector_simulator')
    results = execute(circuit,backend).result().get_counts()

    #Plot just the desired amplitudes
    for i in range(0,(2**n - 2)):
        min_key = min(results.keys(), key=lambda k: results[k]) 
        del results[min_key]
    #To see all the amplitudes - just comment (#) in the for loop above
    
    print("Solution for the input:", input_vector)
    
    answer = list(results.keys())
    
    print('1/sqrt(2) * (|',answer[0],'> + |',answer[1],'>)')
    
    return plot_histogram(results)
            

#To run using a IDE
#just run everything from line 1 to here everything and call the function
#with your desired input
#> sol2_bonus([1,2,3,4...])

            
#To run in the terminal
#python3 Task1_Bonus_Solution1.py [1,2,3,4,...]
if __name__ == "__main__":
    in_vector = sys.argv[1]
    l = list(map(int, in_vector.strip('[]').split(',')))
    sol1_bonus(l)
    
            
            
            
            
        