# https://github.com/ughyper3/grover-quantum-algorithm/blob/5f472987e9b1a900f1d358314c5c7775abdd3028/main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:42:08 2021

@author: dimini
"""

import qiskit 
import numpy as np
from qiskit.visualization import plot_histogram
import random

"""
classical algorithm example and performance : complexity about O(N)
--> N/2 calls to find the element 
"""

def oracle(input):
    winner = 3
    if input is winner:
        response = True
    else:
        response = False
    return response 

def classicalFindingElementAlgorithm():
    my_list = [random.randint(0,100) for i in range(100)] 

    for index, trial_number in enumerate(my_list):
        if oracle(trial_number) is True:
            print('Winner found at index %i'%index)
            print('%i calls to the oracle used'%(index+1))
            print('Ratio between calls and my_list len is :%f'%((index + 1) / len(my_list)))
            break

classicalFindingElementAlgorithm() 


"""
quantum algorithm example and performance : complexity about sqrt(N), quadratic gain 
--> 2^2 states possible with two bits and we can find any states in only 1 call with the oracle
"""

def GroverFunction(winner) :

    try : 
        oracle = qiskit.QuantumCircuit(2, name = "oracle")  # creation of a 2 qubits oracle
        
        if winner == "00": # adapt the oracle in order to find the winner state
            oracle.cx(0, 1)
            oracle.cz(1, 0)
            oracle.cx(0, 1)
            oracle.cx(1, 0)
            oracle.cz(0, 1)
            oracle.cx(1, 0)
            oracle.cz(0, 1)
            
        elif winner == "01":
            oracle.cx(0, 1)
            oracle.cz(1, 0)
            oracle.cx(0, 1)
            
        elif winner == "10": 
            oracle.cx(1, 0)
            oracle.cz(0, 1)
            oracle.cx(1, 0)
            
        elif winner == "11": # state 11
            oracle.cz(1, 0)
        
        else :
            pass
        
            
        oracle.to_gate() 
        oracle.draw() 
        
        backend = qiskit.Aer.get_backend("statevector_simulator")
        grover_circ = qiskit.QuantumCircuit(2, 2, name = "grover") 
        grover_circ.h([0, 1]) 
        grover_circ.append(oracle, [0, 1])
        grover_circ.draw()
        
        
        job = qiskit.execute(grover_circ, backend)
        result = job.result()
        
        sv = result.get_statevector()
        np.around(sv, 2)
        
        reflection = qiskit.QuantumCircuit(2, name = "reflection") 
        reflection.h([0, 1])
        reflection.z([0, 1])
        reflection.cz(0, 1)
        reflection.h([0, 1])
        reflection.to_gate()
        reflection.draw()
        
        
        backend = qiskit.Aer.get_backend("qasm_simulator")
        grover_circ = qiskit.QuantumCircuit(2, 2)
        grover_circ.h([0, 1])
        grover_circ.append(oracle, [0, 1]) # add oracle to my 2 qubits
        grover_circ.append(reflection, [0, 1])
        grover_circ.measure([0, 1], [0, 1])
        grover_circ.draw()
        
        
        job = qiskit.execute(grover_circ, backend, shots=1)
        result = job.result()
        iteration_number = result.get_counts()
        
        results = job.result()
        answer = results.get_counts(grover_circ)
        plot_histogram(answer)
        
        print('%i call(s) to the oracle used to find the state'%iteration_number[winner])

    except KeyError:
            print("Invalid input : please input a state number between 00 and 11")


print("Enter the winner state between 00 and 11")
winner = input() 
GroverFunction(winner)
