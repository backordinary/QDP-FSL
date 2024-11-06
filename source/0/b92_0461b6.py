# https://github.com/gwjacobson/Quantum_Crypto/blob/eed531c9d2edd7d290b1ee42033ca02d66d19e02/B92.py
from qiskit import *
from numpy import pi
import random
import string

#implementation of B92 QKD protocol

sim = Aer.get_backend('qasm_simulator') #local backend to run simulation

bits = [0,1] #classical bits alice can choose

a = [] #initialize alices bit string
for i in range(20): #choose random bit string
    curr_bit = random.choice(bits)
    a.append(curr_bit)

print("Alice's String: "+str(a))

qr = QuantumRegister(20)
cr = ClassicalRegister(20)
qc = QuantumCircuit(qr, cr)#create 8 quibit circuit for 8 bits
for qubit in range(20): #apply basis on each qubit
    if a[qubit] == 1: #diagonal basis on 1 bit
        qc.h(qubit)
    else: #computational basis on 0 bit
        continue

qc.barrier()


a_prime = [] #initial Bobs bit string
for i in range(20):
    curr_bit = random.choice(bits)
    a_prime.append(curr_bit)

print("Bob's String: "+str(a_prime))

for qubit in range(20): #apply basis on each qubit for bob
    if a_prime[qubit] == 1: #diagonal basis on 0 bit
        qc.h(qubit)
    else: #computational basis on 1 bit
        continue

qc.barrier()

qc.measure(qr, cr) #measure all the qubits

job = execute(qc, sim, shots=1, memory=True) #get the results of each meausurment
ket = job.result().get_memory()

print(qc)

print('Measurement Results: '+str(ket)) #our measurement results
ket1 = ket[0] #get the string of bit measurements
m = list(map(lambda i:i, ket1)) #split the string of measurements
m_bits = list(map(int, m)) #turn each bit into an int
m_bits.reverse() #for correct order in alice bits
print(m_bits)

key = [] #our quantum key
for b in range(len(m_bits)):
    if m_bits[b] == 1: #if we measured a 1, use in key
        key.append(a[b]) #use the bit from alice's bit string
    else:
        continue

print('Quantum Key: '+str(key))

while True: 
    message = input("Message to encrypt: ") #message to encrypt
    if len(message) > len(key): #check that message is shorter than key
        print('try a shorter message!')
    else:
        break


alice_int = [ord(mess)-96 for mess in message] #int form of message

encryption = [] # our encrypted message
for i in range(len(alice_int)):
    let = alice_int[i]+key[i] #add key int to message int
    encryption.append(let)

enc_mess = [] #letter version of encryption
for i in range(len(encryption)):
    mul = 1
    n = encryption[i]
    enc_mess.append(chr(n+96)*mul)

enc_mess1 = ''.join(enc_mess) #create encyrpted string
print('Your encrypted message is: '+str(enc_mess1))

bob_int = [ord(i) - 96 for i in enc_mess1] #convert encryption to numbers

bob_mess = []
for i in range(len(bob_int)): #subtract the key from encryption
    let = bob_int[i]-key[i]
    bob_mess.append(let)


decrypt = []
for i in range(len(bob_mess)): #convert decrpted numbers to letters
    mul = 1
    n = bob_mess[i]
    decrypt.append(chr(n+96)*mul)

de_mess = ''.join(decrypt) #decrypted srting
print('Your decrypted message is: '+str(de_mess))
