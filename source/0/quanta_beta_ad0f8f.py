# https://github.com/Madan301/quanta-crypt/blob/073038f9ff3503abed9671dfb24fd798a5ab23e3/quanta_beta.py
from random import randint
import random
from qiskit import QuantumCircuit, Aer, transpile, assemble


def encoder(bits): # function to initiate qubit creation by encoding random bits (user_1 side)
    encoding_bases = []
    message = []
    def random_encoder(detect): # function that encodes (converts) the random binary bits to qubits using random bases
        qc = QuantumCircuit(1, 1)
        if detect == 1:
#            print("1 bit detected")
            qc.initialize([0,1],0)
        elements = ['x', 'h', 'y', 'z']
        chosen = random.choice(elements)
        if chosen == 'x':
            qc.x(0)
            encoding_bases.append('x')
        elif chosen == 'y':
            qc.y(0)
            encoding_bases.append('y')
        elif chosen == 'h':
            qc.h(0)
            encoding_bases.append('h')
        elif chosen == 'z':
            qc.z(0)
            encoding_bases.append('z')
        return qc

    for i in range(len(bits)):
        if bits[i] == 1:
            a = random_encoder(1)
            message.append(a)
        elif bits[i] == 0:
            b = random_encoder(0)
            message.append(b)

#    print("user_1's bases: ",encoding_bases)
 #   print(message)
    return message,encoding_bases


def measure(message): # function to decode the received qubits from user_1 (user_2 side)
    backend = Aer.get_backend('aer_simulator')
    measurements = []
    decoding_bases = []

    def decoder(qbits): # function that measures the received qubits from user_1 using random bases
        elements = ['x', 'y', 'z', 'h']
        chosen = random.choice(elements)
        if chosen == 'x':
            qbits.x(0)
            qbits.measure(0, 0)
            decoding_bases.append('x')
        elif chosen == 'y':
            qbits.y(0)
            qbits.measure(0, 0)
            decoding_bases.append('y')
        elif chosen == 'z':
            qbits.z(0)
            qbits.measure(0, 0)
            decoding_bases.append('z')
        elif chosen == 'h':
            qbits.h(0)
            qbits.measure(0, 0)
            decoding_bases.append('h')
        return qbits
    for i in range(len(message)):
        decoder(message[i])
        aer_sim = Aer.get_backend('aer_simulator')
        qobj = assemble(message[i], shots=1, memory=True)
        result = aer_sim.run(qobj).result()
        measured_bit = int(result.get_memory()[0])
        measurements.append(measured_bit)
#    print("user_2's results: ",measurements)
#    print("user_2's bases: ",decoding_bases)
    return measurements,decoding_bases

# takes the bases used by user_1 to encode and bases from user_2 that was used to decode and matches them to generate key
def key_gen(user_1_bases, user_2_bases, bits):
    good_bits = []
    bits_index = []
    for q in range(len(user_1_bases)):
        if user_1_bases[q] == user_2_bases[q]:
            # If both used the same basis, add
            # this to the list of 'good' bits
            good_bits.append(bits[q])
#    print(good_bits)
    return good_bits


def bit_generator(): # to generate a list of 100 random binary bits
    bits = []
    for i in range(100):
        a = randint(0,1)
        bits.append(a)
    return bits


def base_generator(): # function currently jobless will soon be killed or given a job
    base = []
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    a = randint(0,1)
    bases.append(a)
    return base

# after the key is generated the below function collects random samples from them and compares them to verify the integrity of key
def sampler(user_1_bits,user_2_bits):
    user_1_sample = []
    user_2_sample = []
    index = randint(0,10)
    user_1_sample.append(user_1_bits.pop(index))
    user_2_sample.append(user_2_bits.pop(index))
    index = randint(0,10)
    user_1_sample.append(user_1_bits.pop(index))
    user_2_sample.append(user_2_bits.pop(index))
    index = randint(0,10)
    user_1_sample.append(user_1_bits.pop(index))
    user_2_sample.append(user_2_bits.pop(index))
    index = randint(0,10)
    user_1_sample.append(user_1_bits.pop(index))
    user_2_sample.append(user_2_bits.pop(index))
    index = randint(0,10)
    user_1_sample.append(user_1_bits.pop(index))
    user_2_sample.append(user_2_bits.pop(index))
    return user_1_sample,user_2_sample


print("Key generation has been initiated....")

bits = bit_generator()
#print("user_1 bits: ",bits)
message,encoding_bases = encoder(bits)
results,decoding_bases = measure(message)
user_1_key = key_gen(encoding_bases,decoding_bases,bits)
user_2_key = key_gen(encoding_bases,decoding_bases,results)
print('user-1-key: ',user_1_key)
print('user-2-key: ',user_2_key)
user_1, user_2 = sampler(user_1_key,user_2_key)

if user_1==user_2:
    print("Key verification done")
    print("Key can be used for encrypting and decrypting")
    print("-----------------------------------------------")
    print("-----------------------------------------------")
    print("sample collected from user_1's key: ",user_1)
    print("sample collected from user_2's key: ",user_2)
else:
    print("Key verification failed......")
    print("Key not fit for encryption and decryption.......")
#............................................................................




