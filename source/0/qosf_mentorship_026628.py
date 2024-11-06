# https://github.com/VedDharkar/QOSF_Mentorship_2021/blob/da5ecaba3e8807d72ffccd06de466c2f04e53cee/qosf_mentorship.py
# -*- coding: utf-8 -*-

from qiskit import *
from qiskit import QuantumCircuit, Aer, assemble
import math
from math import pi, log
#import numpy as np
#from qiskit.visualization import plot_histogram, plot_bloch_multivector

#test_list = ['1', '5', '7', '10', '12', '14']
user_list = input("Enter the numbers separated by space:-")
test_list = user_list.split()
# using naive method to
# perform conversion
for i in range(0, len(test_list)):
    test_list[i] = int(test_list[i])
# Printing modified list 
print ("Int list is : " + str(test_list))


def converted_into_bin(test_list1):
    num_list = []
    #Converting test list numbers into binary
    for i in test_list:
        b = format(i, "b")
        num_list.append(b)

    binary_num_list = []

    for item in num_list:
        if len(item) <= len(max(num_list, key=len)):
            item = "0" * (len(max(num_list, key=len)) - len(item)) + str(item)
            binary_num_list.append(item)   
        else:
            continue
    print("Converted to binary:-" + str(binary_num_list))
    #Checking for the condition whether the binaries are adjacent bitstrings of '0' and '1'
    alt_bitstring_list = []
    for Item in binary_num_list:
        for i in range(1, len(Item)):
            if Item[i-1] == Item[i]:
                alt_bitstring_list.append(Item)
            else:
                pass


    dif = list(set(binary_num_list) - set(alt_bitstring_list)) #Selecting required binary representation
    print("The required binaries are:- " + str(dif))
    
    #Getting the selected binaries' indices
    indices = []
    
    for i in range(len(binary_num_list)):
        if binary_num_list[i] in dif:
            indices.append(i)
        else:
            pass
    print("The indices are:-" + str(indices))
    
    #Converting the integer indices into binary
    binary_indices = []
    for i in indices:
        b = format(i, "b")
        binary_indices.append(b)
    
    #Setting their equal length
    binary_indices_eq = []
    for item in binary_indices:
        
        if len(item) <= len(max(binary_indices, key=len)):
            item = "0" * (len(max(binary_indices, key=len)) - len(item)) + str(item)
            binary_indices_eq.append(item)   
        else:
            continue
    print("The binary indices are:- " + str(binary_indices_eq))
    
    #Creating required superposition using Qiskit
    
    qc = QuantumCircuit(len(binary_indices_eq[0]))
    nn = math.log(len(binary_indices_eq), 2)
    nn = int(nn) 
    class ccircuit:
        def __init__(self, control_list):
            self.control_list = control_list
            self.control = control_list[0]
        def mcx(self, n):
            for h in self.control_list:
                if self.control != h:
                    qc.cx(self.control, h)
                else:
                    pass
            #print(qc.draw('text'))
    if (len(binary_indices_eq[0]) == 1):
        qc.h(0)
        print(qc.draw('text'))
    else:
        ones = []
        zeroa = []
        for i in range(len(binary_indices_eq) - 1):
            xored = str(int(binary_indices_eq[i]) ^ int(binary_indices_eq[i+1]))
            if(len(xored) != len(binary_indices_eq[i])):
                xored = "0" * (len(binary_indices_eq[i]) - len(xored)) + str(xored)
            
            xored = xored[::-1]
            
            for p in range(len(xored)):
                if(xored[p] == '1'):
                    ones.append(p)
                elif(xored[p] == '0'):
                    zeroa.append(p)
            
            qc.h(ones[0])
            #hhh = ones[0]
            for item in zeroa:
                if(binary_indices_eq[i][::-1][item] == '1'):
                    qc.x(item)
                else:
                    pass
            #print(zeroa)
            #print(ones)
            if len(ones) > 1:
                v = ccircuit(ones)
                v.mcx(len(ones) - 1)
                
            for a in range(len(ones) -1):
                #if(binary_indices_eq[i][::-1][a] != binary_indices_eq[i+1][::-1][a]):
                p = str(binary_indices_eq[i][::-1][ones[a]]) + str(binary_indices_eq[i+1][::-1][ones[a]])
                q = str(binary_indices_eq[i][::-1][ones[a+1]]) + str(binary_indices_eq[i+1][::-1][ones[a+1]])
                if p == q[::-1]:
                    qc.x(a) 
            
            print(qc.draw('text'))
    
BonusError = 'Length of input list is not of form 2^n, bonus points cancelled!!'

converted_into_bin(user_list)

if(math.modf(math.log(len(test_list), 2))[0] != 0.0):
    print(BonusError)
else:
    print("The input list size is of the form 2^n, so you get bonus points!!!")
    
