# https://github.com/HandyKurniawan/Verona-2049/blob/65ae6b9dbfcbfb10de3254705ab0521c37cbdf48/bb84.py
from qiskit import QuantumCircuit, execute, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from numpy.random import randint
from numpy.random import randint
import numpy as np
import globals
import random
import itertools
from qiskit.providers.aer import QasmSimulator

# This file's code was developed using help from the Qiskit textbook
# Specifically the section on quantum key distribution, please see the link below
# https://www.qmunity.tech/tutorials/quantum-key-distribution-with-bb84

#The function encode_message below, creates a list of QuantumCircuits,
# this takes place before phase 2
def encode_message(bits, bases):
    message = []
    for i in range(len(bits)):
        qc = QuantumCircuit(1,1)
        if bases[i] == 0: # Prepare qubit in Z-basis
            if bits[i] == 0:
                pass
            else:
                qc.x(0)
        else: # Prepare qubit in X-basis
            if bits[i] == 0:
                qc.h(0)
            else:
                qc.x(0)
                qc.h(0)
        qc.barrier()
        message.append(qc)
    return message

# applies corresponding measurement, simulating result of each qubit upon measurement
# this is also used to do interception by eve, which will be decided probabilistically
def measure_message(message, bases):
    backend = Aer.get_backend('aer_simulator')
    measurements = []
    for q in range(len(bases)):
        if bases[q] == 0: # measuring in Z-basis
            message[q].measure(0,0)
        if bases[q] == 1: # measuring in X-basis
            message[q].h(0)
            message[q].measure(0,0)

        aer_sim = Aer.get_backend('aer_simulator')
        qobj = assemble(message[q], shots=1, memory=True)
        result = aer_sim.run(qobj).result()
        measured_bit = int(result.get_memory()[0])
        measurements.append(measured_bit)
    return measurements

# The cipher_encryption and _decryption routines below were made with help
# from the GitHub user VoxelPixel, link below:
# https://github.com/VoxelPixel/CiphersInPython/blob/master/XOR%20Cipher.py
def cipher_encryption(msg,key):
    encrypt_hex = ""
    key_itr = 0
    for i in range(len(msg)):
        temp = ord(msg[i]) ^ ord(key[key_itr])
        encrypt_hex += hex(temp)[2:].zfill(2)
        key_itr += 1
        if key_itr >= len(key):
            key_itr = 0
    return format(encrypt_hex)

def cipher_decryption(msg,key):
    hex_to_uni = ""
    for i in range(0, len(msg), 2):
        hex_to_uni += bytes.fromhex(msg[i:i+2]).decode('utf-8')
    decryp_text = ""
    key_itr = 0
    for i in range(len(hex_to_uni)):
        temp = ord(hex_to_uni[i]) ^ ord(key[key_itr])
        decryp_text += chr(temp)
        key_itr += 1
        if key_itr >= len(key):
            key_itr = 0
    return format(decryp_text)

# sifting stage.
def sift(a_bases, b_bases, bits):
    good_bits = []
    for q in range(globals.selectedBit):
        if a_bases[q] == b_bases[q]:
            good_bits.append(bits[q])
    return good_bits


# phase 4; comparing random select of bits in their keys
def sample_bits(bits, selection):
    sample = []
    for i in selection:
        i = np.mod(i, len(bits))
        sample.append(bits.pop(i))
    return sample

