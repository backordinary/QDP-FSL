# https://github.com/kiranyak/Quanta-Security-/blob/fef932b5af16b07ac951bc962b2e13ce3c302c12/without_Eve_The_Hacker.py
#!/usr/bin/env python3
#export PYTHONUNBUFFERED=1
# coding: utf-8



import qiskit
#qiskit.__qiskit_version__

#other useful packages
import math
import matplotlib.pyplot as plt
import numpy as np

# Import Qiskit
from qiskit import BasicAer, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from time import sleep

CRED    = '\033[31m'
CGREEN  = '\033[32m'
CYELLOW = '\033[33m'
CBLUE   = '\033[34m'
CVIOLET = '\033[35m'
CEND = '\033[0m'

#Super secret message
print('')
print('')
print(CGREEN + "******" + CEND + CYELLOW + " Starting Quantum Program " + CEND + CGREEN + "******" + CEND )
print('')
print('')
mes = input( CVIOLET + "Enter your secret message: " + CVIOLET)
print('')
print('')
sleep(1)

#initial size of key
n = len(mes)*3

#break up message into smaller parts if length > 10
nlist = []
for i in range(int(n/10)):
    nlist.append(10)
if n%10 != 0:
    nlist.append(n%10)

print(CGREEN +'Initial key length: ' + CEND ,n )

# Make random strings of length string_length

def randomStringGen(string_length):
    #output variables used to access quantum computer results at the end of the function
    output_list = []
    output = ''
    
    #start up your quantum circuit information
    backend = BasicAer.get_backend('qasm_simulator')  
    circuits = ['rs']
    
    #run circuit in batches of 10 qubits for fastest results. The results
    #from each run will be appended and then clipped down to the right n size.
    n = string_length
    temp_n = 10
    temp_output = ''
    for i in range(math.ceil(n/temp_n)):
        #initialize quantum registers for circuit
        q = QuantumRegister(temp_n, name='q')
        c = ClassicalRegister(temp_n, name='c')
        rs = QuantumCircuit(q, c, name='rs')
            
        #create temp_n number of qubits all in superpositions
        for i in range(temp_n):
            rs.h(q[i]) #the .h gate is the Hadamard gate that makes superpositions
            rs.measure(q[i],c[i])

        #execute circuit and extract 0s and 1s from key
        result = execute(rs, backend, shots=1).result()
        counts = result.get_counts(rs)
        result_key = list(result.get_counts(rs).keys())
        temp_output = result_key[0]
        output += temp_output
        
    #return output clipped to size of desired string length
    return output[:n]


key = randomStringGen(n)
print('')
print(CGREEN + "******" + CEND + CYELLOW + " generate random rotation strings for Alice and Bob " + CEND + CGREEN + "******" + CEND )
print('')
print(CGREEN + 'Initial key: ' + CEND ,key, flush = True)

#generate random rotation strings for Alice and Bob
Alice_rotate = randomStringGen(n)
Bob_rotate = randomStringGen(n)
print(CGREEN + "Alice's rotation string:" + CEND,Alice_rotate, flush = True)

sleep(3)
print(CYELLOW + "Bob's rotation string:  " + CEND,Bob_rotate, flush = True)

#start up your quantum program
print('')
print(CGREEN + "******" + CEND + CYELLOW + " Starting up " + CEND + CGREEN + "******" + CEND )
print('')

backend = BasicAer.get_backend('qasm_simulator')  
shots = 1
circuits = ['send_over']
Bob_result = ''

for ind,l in enumerate(nlist):
    #define temp variables used in breaking up quantum program if message length > 10
    if l < 10:
        key_temp = key[10*ind:10*ind+l]
        Ar_temp = Alice_rotate[10*ind:10*ind+l]
        Br_temp = Bob_rotate[10*ind:10*ind+l]
    else:
        key_temp = key[l*ind:l*(ind+1)]
        Ar_temp = Alice_rotate[l*ind:l*(ind+1)]
        Br_temp = Bob_rotate[l*ind:l*(ind+1)]
    
    #start up the rest of your quantum circuit information
    q = QuantumRegister(l, name='q')
    c = ClassicalRegister(l, name='c')
    send_over = QuantumCircuit(q, c, name='send_over')
    
    #prepare qubits based on key; add Hadamard gates based on Alice's and Bob's
    #rotation strings
    for i,j,k,n in zip(key_temp,Ar_temp,Br_temp,range(0,len(key_temp))):
        i = int(i)
        j = int(j)
        k = int(k)
        if i > 0:
            send_over.x(q[n])
        #Look at Alice's rotation string
        if j > 0:
            send_over.h(q[n])
        #Look at Bob's rotation string
        if k > 0:
            send_over.h(q[n])
        send_over.measure(q[n],c[n])

    #execute quantum circuit
    result_so = execute([send_over], backend, shots=shots).result()
    counts_so = result_so.get_counts(send_over)
    result_key_so = list(result_so.get_counts(send_over).keys())
    Bob_result += result_key_so[0][::-1]

sleep(3)
print(CYELLOW + "Bob's results: " + CEND, Bob_result, flush = True)


def makeKey(rotation1,rotation2,results):
    key = ''
    count = 0
    for i,j in zip(rotation1,rotation2):
        if i == j:
            key += results[count]
        count += 1
    return key
  
Akey = makeKey(Bob_rotate,Alice_rotate,key)
Bkey = makeKey(Bob_rotate,Alice_rotate,Bob_result)

sleep(2)
print( CGREEN + "Alice's key:" + CEND,Akey, flush = True)
print( CYELLOW + "Bob's key:  " + CEND,Bkey, flush = True)


#make key same length has message
shortened_Akey = Akey[:len(mes)]
encoded_m=''

#encrypt message mes using encryption key final_key
for m,k in zip(mes,shortened_Akey):
    encoded_c = chr(ord(m) + 2*ord(k) % 256)
    encoded_m += encoded_c

sleep(2)
print('')
print(CGREEN + "******" + CEND + CYELLOW + " Encrypting the message using encryption key" + CEND + CGREEN + "******" + CEND )
print('')
print(CGREEN + 'encoded message:  ' + CEND,encoded_m.encode("utf-8"), flush = True)

#make key same length has message
shortened_Bkey = Bkey[:len(mes)]


#decrypt message mes using encryption key final_key
result = ''
for m,k in zip(encoded_m,shortened_Bkey):
    encoded_c = chr(ord(m) - 2*ord(k) % 256)
    result += encoded_c

sleep(2) 
print('')
print(CGREEN + "******" + CEND + CYELLOW + "Decrypting the message " + CEND + CGREEN + "******" + CEND )
print('')
print('')
print(CYELLOW + 'Message received by Bob:' + CEND,result, flush = True)
print('')
#start up your quantum program
backend = BasicAer.get_backend('qasm_simulator')  
shots = 1
circuits = ['Eve']

Eve_result = ''
for ind,l in enumerate(nlist):
    #define temp variables used in breaking up quantum program if message length > 10
    if l < 10:
        key_temp = key[10*ind:10*ind+l]
        Ar_temp = Alice_rotate[10*ind:10*ind+l]
    else:
        key_temp = key[l*ind:l*(ind+1)]
        Ar_temp = Alice_rotate[l*ind:l*(ind+1)]
    
    #start up the rest of your quantum circuit information
    q = QuantumRegister(l, name='q')
    c = ClassicalRegister(l, name='c')
    Eve = QuantumCircuit(q, c, name='Eve')
    
    #prepare qubits based on key; add Hadamard gates based on Alice's and Bob's
    #rotation strings
    for i,j,n in zip(key_temp,Ar_temp,range(0,len(key_temp))):
        i = int(i)
        j = int(j)
        if i > 0:
            Eve.x(q[n])
        if j > 0:
            Eve.h(q[n])
        Eve.measure(q[n],c[n])
    
    #execute
    result_eve = execute(Eve, backend, shots=shots).result()
    counts_eve = result_eve.get_counts()
    result_key_eve = list(result_eve.get_counts().keys())
    Eve_result += result_key_eve[0][::-1]

#print("Eve's results: ", Eve_result)#start up your quantum program
backend = BasicAer.get_backend('qasm_simulator')  
shots = 1
circuits = ['Eve2']

Bob_badresult = ''
for ind,l in enumerate(nlist):
    #define temp variables used in breaking up quantum program if message length > 10
    if l < 10:
        key_temp = key[10*ind:10*ind+l]
        Eve_temp = Eve_result[10*ind:10*ind+l]
        Br_temp = Bob_rotate[10*ind:10*ind+l]
    else:
        key_temp = key[l*ind:l*(ind+1)]
        Eve_temp = Eve_result[l*ind:l*(ind+1)]
        Br_temp = Bob_rotate[l*ind:l*(ind+1)]
    
    #start up the rest of your quantum circuit information
    q = QuantumRegister(l, name='q')
    c = ClassicalRegister(l, name='c')
    Eve2 = QuantumCircuit(q , c, name='Eve2')
    
    #prepare qubits
    for i,j,n in zip(Eve_temp,Br_temp,range(0,len(key_temp))):
        i = int(i)
        j = int(j)
        if i > 0:
            Eve2.x(q[n])
        if j > 0:
            Eve2.h(q[n])
        Eve2.measure(q[n],c[n])
    
    #execute
    result_eve = execute(Eve2, backend, shots=shots).result()
    counts_eve = result_eve.get_counts()
    result_key_eve = list(result_eve.get_counts().keys())
    Bob_badresult += result_key_eve[0][::-1]
    
#print("Bob's previous results (w/o Eve):",Bob_result)
#print("Bob's results from Eve:\t\t ",Bob_badresult)

#make keys for Alice and Bob
Akey = makeKey(Bob_rotate,Alice_rotate,key)
Bkey = makeKey(Bob_rotate,Alice_rotate,Bob_result)
#print("Alice's key:   ",Akey)
#print("Bob's key:     ",Bkey)

check_key = randomStringGen(len(Akey))
#print('spots to check:',check_key)

#find which values in rotation string were used to make the key
Alice_keyrotate = makeKey(Bob_rotate,Alice_rotate,Alice_rotate)
Bob_keyrotate = makeKey(Bob_rotate,Alice_rotate,Bob_rotate)

# Detect Eve's interference
#extract a subset of Alice's key
sub_Akey = ''
sub_Arotate = ''
count = 0
for i,j in zip(Alice_rotate,Akey):
    if int(check_key[count]) == 1:
        sub_Akey += Akey[count]
        sub_Arotate += Alice_keyrotate[count]
    count += 1
print('')
print(CGREEN + "******" + CEND + CYELLOW + "Detecting the presence of Eve" + CEND + CGREEN + "******" + CEND )
print('')
#extract a subset of Bob's key
sub_Bkey = ''
sub_Brotate = ''
count = 0
for i,j in zip(Bob_rotate,Bkey):
    if int(check_key[count]) == 1:
        sub_Bkey += Bkey[count]
        sub_Brotate += Bob_keyrotate[count]
    count += 1

sleep(3)
print(CGREEN + "subset of Alice's key:" + CEND, sub_Akey, flush = True)
print(CYELLOW + "subset of Bob's key:  " + CEND ,sub_Bkey, flush = True)
print('')
print('')

#compare Alice and Bob's key subsets
print('')
print(CGREEN + "******" + CEND + CYELLOW + "Comparing Alice and Bob's key subsets " + CEND + CGREEN + "******" + CEND )
print('')
sleep(5)
secure = True
for i,j in zip(sub_Akey,sub_Bkey):
    if i == j:
        secure = True
    else:
        secure = False
        break;
if not secure:
    print(CRED + 'Eve detected!' + CRED, flush = True)
else:
    print(CGREEN + 'No Eve-The Hacker Present in your communication Channel.!' + CEND, flush = True)

#sub_Akey and sub_Bkey are public knowledge now, so we remove them from Akey and Bkey
#if secure:
    #print("Restart the program for sending new message!! ", flush = True)

sleep(1)
print('')
print('')
print(CGREEN + "******" + CEND + CYELLOW + " Ending Quantum Program !! Thank you!! " + CEND + CGREEN + "******" + CEND, flush = True )
sleep(1)