# https://github.com/psasanka1729/Grover-NISQ/blob/a410b3273db970396ab0b9ac31cd0cd5b0b1eec4/nisq-grover-main/grover_gates.py
'''
This code generates a txt file of the decomposition of the Grover operator with
the target state as input.

'''



#!/usr/bin/env python
# coding: utf-8

# In[32]:

import sys
from qiskit import*
#from qiskit import Aer
import qiskit.quantum_info as qi
import numpy as np
#from math import pi
#import matplotlib.pyplot as plt
import re
#import csv
#import time
#start = time.time()


# ### Target state $|w>$

# In[33]:

Target_state = '00000000'


# ## Setting up the The Grover operator
# $$ G = U_w U_s$$
# Where 
# $$U_w = 2|w><w| - I$$ 
# $U_w$ is a matrix with the position of the target state 1 and all its diagnoal elements are -1 and rest all zero.

# In[34]:


# First we note the length of N.
N = len(Target_state)

# Then an identity matrix is created with the size 2**N with signs reversed.
U_w = - np.identity(2 ** N, dtype=complex)


# Then the sign of the element corresponding to the target state is flipped. To do that we first convert the
# target state from binary to decimal number. 
Target_index = int(Target_state, 2)

## The position of the target state is set as 1.
U_w.itemset((Target_index, Target_index),1)


# ## Setting up the circuit

# In[36]:


qc = QuantumCircuit(N)
qc.unitary(U_w,[i for i in range(N)])
#qc.draw('mpl')


# ## For transpile, we will choose the basis gates as $Rz, CNOT$ and $H$

# In[37]:


trans_qc = transpile(qc, basis_gates = ['id', 'rz', 'cx','h'], optimization_level = 1)
#trans_qc.draw('mpl')


# ### Writing $1$ and $2$ qubit gates as human readable format.
# The following loop puts the gates in a text file.

# In[38]:


## (Instruction(name='rz', num_qubits=1, num_clbits=0, params=[0.9817477042468101]),
#[Qubit(QuantumRegister(3, 'q'), 0)], [])

## (Instruction(name='h', num_qubits=1, num_clbits=0, params=[]), [Qubit(QuantumRegister(3, 'q'), 0)], [])

## (Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]),
#[Qubit(QuantumRegister(3, 'q'), 1), Qubit(QuantumRegister(3, 'q'), 0)], [])

f = trans_qc
def List_Gates(Gate_Number):
    
    
    ## The list will hold string along with the angles in case of u3.
    l = []
    
    Name_of_Gate = []
    
    ## The list will hold the qubit in case of u3.
    Qubit = []    
    
    ## The list will hold the numerical value (float) of the angle in case of u3.
    Angle = []

    ## The following two lines will be used to separate the numerical value of the angle from 
    # a mixture of string and digits.
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ )?'
    
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    
    if str(trans_qc[Gate_Number][0]).split(',')[0][-3:-1] == 'rz':
        
        Name_of_Gate.append('rz')        
        ## For the gate rz, there is one angle. The following lines puts the string which contain
        # the angle along with other alphabets in a list l.
        l.append(str(list(f)[Gate_Number]).split(',')[3])

        
        ## The following loop extracts the angle from the string of alphabet and digit.
        for i in l:
            
            if rx.findall(i) == []:
                
                ## the angle zero is written as 0.0 .
                Angle.append(0.0)
                
            else:
                
                Angle.append(float(rx.findall(i)[0]))
                
        ## the qubit on which rz acts is listed.        
        Qubit.append(re.sub("\D", "", str(trans_qc[Gate_Number][1]).split(',')[2]))  
        
    # For hadamard gates the qubit it is applied to is listed.    
    elif str(trans_qc[Gate_Number][0]).split(',')[0][-2:-1] == 'h':
        
        Name_of_Gate.append('h')
        Qubit.append(re.sub("\D", "", str(trans_qc[Gate_Number][1]).split(',')[2]))
    
    
        ## if the gate is cx then the Angle list is empty.        
    else:
        
        Name_of_Gate.append('cx')     
        ## the control and the target bit is listed.
        Control_bit = re.sub("\D", '', str(trans_qc[Gate_Number][1][0]).split(',')[2])
        
        Target_bit = re.sub("\D", '', str(trans_qc[Gate_Number][1][1]).split(',')[2])   
        
        Qubit.append([Control_bit, Target_bit])
    
    
    return Name_of_Gate,Angle, Qubit



# The following list has all the gates in the format [name of the gate, angle, qubit].
l = []
for i in range(len(f)):
    l.append(List_Gates(i))      

## To view the gates, open the following text file.    
file = open('gates_list'+Target_state+'.txt', 'w')


## The following loop writes the gates into the file.
for i in range(len(l)):
    
    if l[i][0][0] == 'rz':
        
        file.write('rz'+','+str(l[i][1][0])+','+str(l[i][2][0])+','+'\n')

        
    elif l[i][0][0] == 'h':
        
        file.write('h'+','+'0.0'+','+str(l[i][2][0])+','+'\n')
        
        
    else:
        
        file.write('cx'+','+l[i][2][0][0]+','+l[i][2][0][1]+','+'\n')
