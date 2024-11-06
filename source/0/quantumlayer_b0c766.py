# https://github.com/Tovermodus/Tic-Tac-Qu-Learning/blob/5e292809f95370ba858e7f99adf184a6ff1f19c6/python/learning/quantumlayer.py
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:59:55 2020

@author: nvemonds
"""

import numpy as np
from qiskit import *
from qiskit.visualization import *
from qiskit.visualization import plot_histogram
#%%

class activation_function:

    def __init__(self, secondLayerNodes, func):
        
        self.theta = circuit.Parameter('theta')
        self.secondLayerNodes = secondLayerNodes
        
        all_qubits=range(secondLayerNodes)
        
        self.qc = QuantumCircuit(secondLayerNodes, secondLayerNodes)
        self.qc.h(all_qubits)
        self.qc.barrier()
        self.qc.ry(self.theta, all_qubits)        
        self.qc.measure(all_qubits, all_qubits)
        
        self.func=func
        
    def run(self, input_vector,simulator, shots):
        """
            maps values of tuples (length=secondLayerNodes) to values 0<x<1

        """
        thetas=[self.func(input_vector)]
        job = execute(self.qc, simulator, shots = shots,  parameter_binds = [{self.theta: theta} for theta in thetas])
        
        result = job.result().get_counts(self.qc)
        counts = np.array(list(result.values()))
        probabilities = counts / shots
        states = np.array(list(result.keys()))

        # sum up all probabilities that imply, that qubit k collapses to 1
        results =[sum([probabilities[i] for i in range(len(probabilities)) if states[i][k]=="1"]) for k in range(self.secondLayerNodes)]
        return results
    
#%%
simulator = Aer.get_backend('qasm_simulator')

test=activation_function(7, lambda x: np.linalg.norm(x)*np.pi)
print(test.run([0.4,0.7,0.2,0.9],simulator, 1000))