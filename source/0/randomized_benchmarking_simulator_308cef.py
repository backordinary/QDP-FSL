# https://github.com/NTHU-SQC/Quantum-Compiler/blob/84e74abb4071d94c2b9a9739ac8902da3703e6b1/Quantum%20Circuits/Randomized_Benchmarking_simulator/Randomized_Benchmarking_simulator.py
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import random
import numpy as np
import copy
import math as mt
import cmath
from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt
from IPython import display
from scipy import optimize

### State Initialization

init_state = np.array([1/mt.sqrt(2),1/mt.sqrt(2)])
final_state = np.array([1/mt.sqrt(2),1/mt.sqrt(2)])
final_state = np.transpose(final_state)

### Gate Initialization
I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,1j],[-1j,0]])
Z = np.array([[1,0],[0,-1]])
H = np.array([[1/mt.sqrt(2),1/mt.sqrt(2)],[1/mt.sqrt(2),-1/mt.sqrt(2)]])
S = np.array([[1,0],[0,1j]])

#U = np.array()


Pauli_g = {"I":I, "X":X, "Y":Y, "Z":Z}
Pauli_list = ["I", "X", "Y", "Z"]

Clifford_g = {"H":H, "S":S}
Clifford_list = ["H","S"]


### Error Simulation
d_step = 0.999
eps_step = 0
step_err = np.array([[d_step,eps_step],[eps_step,d_step]])

d_gate = 0.989
eps_gate = 0
gate_err = np.array([[d_gate,eps_gate],[eps_gate,d_gate]])

#print(H)
#print(S)
#print(I.shape)
#print(Clifford_list[0])
#print(np.matmul(Y,X))

Gate_Interest = copy.deepcopy(H)





### Randomized Benchmarking Protocol


#seq_truncated_list = [2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96]
seq_truncated_list =  [2,3,10,30,50,70,100,150,200,250,300,350,400,450,500]
#print(seq_truncated_list)
average_sigmaz_list = []


for num in range(len(seq_truncated_list)):

    #num = 3
    m = seq_truncated_list[num]
    
    
    
    num_shot = 100
    sum_sigmaz = 0
    
    for shot in range(num_shot):
        
        Fin = copy.deepcopy(I)
        recover_Fin = copy.deepcopy(I)

        Fin_interleaved = copy.deepcopy(I)
        recover_Fin_interleaved = copy.deepcopy(I)
    
    
    ### Standard 
        """
        for idx in range(m):
            
            #k1 = random.randint(1,4)
            k2 = random.randint(1,2)
            #c = Pauli_list[k1-1]
            #cgate = Pauli_g[c]
            #print(k)
            c = Clifford_list[k2-1]
            cgate = Clifford_g[c]
            print(c)
            print(cgate)
            Fin = np.matmul(cgate,Fin)
            
        #print(count)
        
        
        print("The result of the standard RB matrix multiplication:" + "\n" +str(Fin))
        
        recover_Fin = np.linalg.inv(Fin)
        verify  = np.matmul(recover_Fin,Fin)
        print("Standard RB Verification:" + "\n" + str(verify))
        """
        ### Interleaved
        
        for idx in range(m):
            #k3 = random.randint(1,4)
            k4  = random.randint(1,2)
            #c  = Pauli_list[k3-1]
            #cgate = Pauli_g[c]
            c_2 = Clifford_list[k4-1]
            cgate_interleaved = Clifford_g[c_2]
            print(c_2)
            Fin_interleaved = np.matmul(cgate_interleaved,Fin_interleaved)
            Fin_interleaved = np.matmul(step_err,Fin_interleaved)
            Fin_interleaved = np.matmul(gate_err,Fin_interleaved)
            Fin_interleaved = np.matmul(Gate_Interest,Fin_interleaved)
            recover_Fin_interleaved = np.matmul(cgate_interleaved,recover_Fin_interleaved)
            recover_Fin_interleaved = np.matmul(Gate_Interest,recover_Fin_interleaved)
            print(Fin_interleaved)
            print(recover_Fin_interleaved)
            print("\n=================")
            
            
        print("The result of the interleaved RB matrix multiplication:" + "\n" + str(Fin_interleaved))
        
        recover_Fin_interleaved_  = np.matmul(step_err,recover_Fin_interleaved)
        recover_Fin_interleaved_f = np.linalg.inv(recover_Fin_interleaved_)
    
        result = np.matmul(recover_Fin_interleaved_f,Fin_interleaved)
        
        print("Interleaved RB Verification:" + "\n" + str(result))
        
        print("\n++++++= projection =+++++++")
        
        sigma_z = np.matmul(init_state,result)
        print(sigma_z)
        
        print("\n++++++= expectation value =+++++++")
        sigma_z = np.matmul(sigma_z,final_state)
        print("<sigma_z> :" + str(sigma_z))
        
        
        sum_sigmaz = sum_sigmaz + sigma_z
        
        
    print("\n++++++= average over shots =+++++++")
    average_sigmaz = np.abs(sum_sigmaz/num_shot)
    
    print("The average value of sigma z :" + "\n" + str(average_sigmaz))
    
    average_sigmaz_list.append(average_sigmaz)
    
#print(average_sigmaz_list)

### Make Plots 

plt.scatter(seq_truncated_list,average_sigmaz_list)
plt.xlabel("Length of random sequence")
plt.ylabel("Average Fidelity")
plt.text(300,0.95,f'Shot number : {num_shot}',)
plt.text(300,0.90,f'Clifford error : {d_step}',)
plt.text(300,0.85,f'gate error : {d_gate}',)


#z = np.polyfit(seq_truncated_list,average_sigmaz_list,len(seq_truncated_list))
#def test_func(x, a, m, b):
    #return a*x**m +b


#params, params_covariance = optimize.curve_fit(test_func, seq_truncated_list, average_sigmaz_list,p0=[1,len(seq_truncated_list),1])
#plt.plot(seq_truncated_list, test_func(seq_truncated_list, params[0], params[1],params[2]),label='Fitted function')
#plt.show()


    
    