# https://github.com/mayylinh/Quantum-Key/blob/18d1b854bd4f837c50ea4c53ef508731160827f8/QuantumKey.py
#!/usr/bin/env python
# coding: utf-8

############################################################
# Alice's Knowledge | Over Eve's Channel | Bob's Knowledge #
############################################################
# alice_bits        |                    |                 #
# alice_bases       |                    |                 #
# message           | message            | message         #
#                   |                    | bob_bases       #
#                   |                    | bob_results     #
#                   | alice_bases        | alice_bases     #
# bob_bases         | bob_bases          |                 #
# alice_key         |                    | bob_key         #
# bob_sample        | bob_sample         | bob_sample      #
# alice_sample      | alice_sample       | alice_sample    #
# shared_key        |                    | shared_key      #
############################################################




get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()

from qiskit.visualization import plot_histogram, plot_bloch_multivector
from numpy.random import randint
import numpy as np

np.random.seed(0)

bitsLen = 100

#Step 1: Alice randomly generates 100 bits
alice_bits = randint(2, size=bitsLen)
print("Alice's bits: " + str(alice_bits))


#Step 2: Create an array of qubit bases
# 0 or 1 represent bases
# Z- or X-
# on which a qubit is encoded, respectively
alice_bases = randint(2, size=bitsLen)
print("Alice's bases: " + str(alice_bases))


#Function to encode message based on qubit
def encode_message(bits, bases):
    message = []
    for i in range(bitsLen):
        qc = QuantumCircuit(1,1)
        #Prepare qubit in Z-basis
        if bases[i] == 0:
            if bits[i] == 0:
                pass
            else:
                qc.x(0)
        #Prepare qubit in X-basis
        else:
            if bits[i] == 0:
                qc.h(0)
            else:
                qc.x(0)
                qc.h(0)
        qc.barrier()
        message.append(qc)
    return message


#Step 3: Encode Alice's 100 bits based on encoding bases
message = encode_message(alice_bits, alice_bases)

print('bit = %i' % alice_bits[0])
print('basis = %i' % alice_bases[0])

message[0].draw()


#Function to apply recipient's bases and simualte qubit measurement results
def measure_message(message, bases):
    backend = Aer.get_backend('qasm_simulator')
    measurements = []
    for q in range(bitsLen):
        #measuring in Z-basis
        if bases[q] == 0:
            message[q].measure(0,0)
        #measuring in X-basis
        if bases[q] == 1:
            message[q].h(0)
            message[q].measure(0,0)
        result = execute(message[q], backend, shots=1, memory=True).result()
        measured_bit = int(result.get_memory()[0])
        measurements.append(measured_bit)
    return measurements


#Step 4: Bob decides which basis to measure in
bob_bases = randint(2, size=bitsLen)

print("Bob's bases: " + str(bob_bases))


#Step 5: Bob measures each qubit in message with his bases
bob_results = measure_message(message, bob_bases)

print("Bob's measurement results: " + str(bob_results))
message[0].draw()


#For every basis in bob_bases that matched alice_bases, 
# an entry in bob_results will match its corresponding entry in alice_bits
# matching bits will used as part of their key

#Function to discard Bob's random results from different bases
def remove_garbage(a_bases, b_bases, bits):
    key_bits = []
    for q in range(bitsLen):
        #If both used the same basis, bit is added to key_bits
        if a_bases[q] == b_bases[q]:
            key_bits.append(bits[q])
    return key_bits


#Step 6: Alice and Bob create their keys by calling remove_garbage
alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
bob_key = remove_garbage(alice_bases, bob_bases, bob_results)

print("Alice's key: " + str(alice_key))
print("\nBob's key: " + str(bob_key))


#Function to compare random selection of bits in keys 
# to make sure key generation worked correctly
# selected bits are removed from key as they are publicly broacasted
# and no longer secret
def sample_bits(bits, selection):
    sample = []
    for i in selection:
        i = np.mod(i, len(bits))
        sample.append(bits.pop(i))
    return sample


#Step 7: Test to see if keygen worked and remove sample bits
sample_size = 15
bit_selection = randint(bitsLen, size=sample_size)

alice_sample = sample_bits(alice_key, bit_selection)
bob_sample = sample_bits(bob_key, bit_selection)

print("Alice's sample: " + str(alice_sample))
print("\nBob's sample: " + str(bob_sample))
print("\nbob_sample =?= alice_sample ")
print(bob_sample == alice_sample)


#Alice and Bob both have the same shared key
print("Alice's key: " + str(alice_key))
print("\nBob's key: " + str(bob_key))
print("\nKey length = %i" % len(alice_key))
