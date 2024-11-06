# https://github.com/iamleoluo/Application-of-QKD-in-Image-Transmission/blob/556409f10697d4a79b58910b306cd178b99b08f5/importQKD.py
import numpy as np
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# from ibm_quantum_widgets import *
from random import getrandbits
import binascii

'''def module(ibmapiToken):
    from qiskit import QuantumCircuit, execute, transpile, Aer, IBMQ
    from qiskit.tools.jupyter import *
    from qiskit.visualization import *
    from ibm_quantum_widgets import *
    from random import getrandbits
    #from qiskit_textbook.tools import array_to_latex
    # Loading your IBM Quantum account(s)
    provider = IBMQ.enable_account(ibmapiToken)
import_module('dd8c461ba86616beebb9455030bf20f149e4882b8d27cd3d1e40f0d65e0a2b1d4dc5f90591e13d72b3bfcd583cb3f14154882a7afa58918e7cda3a95f4aaf191')
'''


def AliceChooseBits(AmountofQubit):  # Generate Alice's bits.
    print('AliceChooseBits')
    alice_bits = []  # This list will store Alice's bits
    for i in range(AmountofQubit):  # Generate qubits random bits
        alice_bits.append(str(getrandbits(1)))  # The function getrandbits generates 1 random bit
    return alice_bits


def AliceChooseBases(AmountofQubit):
    print("AliceChooseBases")
    # Generate Alice's bases.
    alice_bases = []  # List to store Alice's bases
    for i in range(AmountofQubit):
        base = getrandbits(1)
        if base == 0:
            alice_bases.append("Z")
        else:
            alice_bases.append("X")
    return alice_bases


def EncodeClassicaltoQubits(alice_bases, alice_bits, AmountofQubit):
    # Encode Alice's qubits. Remeber that the qubit will be in the |0> state at the start, before any gates are applied.
    encoded_qubits = []
    for i in range(AmountofQubit):
        qc = QuantumCircuit(1, 1)
        if alice_bases[i] == "Z":
            if alice_bits[i] == '0':
                pass  # We want nothing to happen here - the qubit is already in the state |0>
            elif alice_bits[i] == '1':
                qc.x(0)  # Applying an X gate to change the qubit state to |1>
        elif alice_bases[i] == "X":
            if alice_bits[i] == '0':
                qc.h(0)  # Applying an H gate to change the qubit state to |+>
            elif alice_bits[i] == '1':
                qc.x(0)  # Applying an X and H gate to change the qubit state to |->
                qc.h(0)
        encoded_qubits.append(
            qc)  # Adding the qubit with the right state to the list of qubits that Alice will send Bob
        #print(str(i) + "bits encoded")
    print("encoded all classical bits to qubits ")
    return encoded_qubits


def BobChooseBases(AmountofQubit):
    print("BobChooseBases")
    # generate Bob's bases
    bob_bases = []
    for i in range(AmountofQubit):
        bit = getrandbits(1)
        if bit == 0:
            bob_bases.append("Z")
        else:
            bob_bases.append("X")
    return bob_bases


def BobMeasure(bob_bases, encoded_qubits, AmountofQubit):
    print('BobMeasure')
    # measure Alice's qubits. Remeber that if a bit bob_bases is 0, Bob measures in the Z basis. If the bit is 1,
    bob_bits = []  # List of Bob's bits generated from the results of Bob's measurements
    for i in range(AmountofQubit):
        qc = encoded_qubits[i]
        if bob_bases[i] == "Z":  # Bob's basis is Z
            qc.measure(0, 0)  # Code to measure in the Z basis
        elif bob_bases[i] == "X":  # The case that bob_bases[i] is X
            qc.h(
                0)  # WRITE CODE HERE: Bob needs to add a gate here to correctly measure in the X basis. Which gate is this?
            qc.measure(0, 0)  # Measurement in the X basis
        # Now that the measurements have been added to the circuit, let's run them.
        job = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=1)
        results = job.result()
        counts = results.get_counts()
        measured_bit = max(counts, key=counts.get)
        #print(str(i) + "qubits measured")

        # Append measured bit to Bob's measured bitstring
        bob_bits.append(measured_bit)
    return bob_bits


def CompareBases(alice_bases, bob_bases):
    print("CompareBases")
    # BLOCK 9 - Alice and Bob compare their bases
    agreeing_indices = []
    for i in range(len(alice_bases)):
        if alice_bases[i] == bob_bases[i]:
            agreeing_indices.append(
                i)  # This statement will print how many bases were the same for Alice and Bob. Can you guess what this number should approximately be?
    return agreeing_indices


def AliceCreateKey(agreeing_indices, alice_bits):
    alice_key = []
    for index in agreeing_indices:
        alice_key.append(alice_bits[index])

    # WRITE CODE HERE: Add a print statement to see Alice's key
    return alice_key


def BobCreateKey(agreeing_indices, bob_bits):
    bob_key = []
    for index in agreeing_indices:
        bob_key.append(bob_bits[index])
    # WRITE CODE HERE: Add a print staement to see Bob's key
    return bob_key


def encrypt_message(ImgBits, key):
    # bitstring = ImgBits.zfill(8 * ((len(ImgBits) + 7) // 8))
    bitstring = ImgBits
    # created the encrypted string using the key
    encrypted_string = ""
    lenKey = len(key)
    lenBitstring = len(bitstring)
    '''if len(bitstring) > lenKey:
        for i in range(len(bitstring)//lenKey):
            for j in range(lenKey):
                key.append(key[j])'''
    for i in range(lenBitstring):
        encrypted_string += str((int(bitstring[i]) ^ int(key[i - lenKey * (i // lenKey)])))
    print("encrypted")
    return encrypted_string


def decrypt_message(encrypted_bits, key):
    # created the unencrypted string using the key
    unencrypted_bits = ""
    lenKey = len(key)
    for i in range(len(encrypted_bits)):
        unencrypted_bits += str((int(encrypted_bits[i]) ^ int(key[i - lenKey * (i // lenKey)])))
    # Convert bitstring into
    i = int(unencrypted_bits, 2)
    hex_string = '%x' % i
    n = len(hex_string)
    bits = binascii.unhexlify(hex_string.zfill(n + (n & 1)))
    # unencrypted_string = bits.decode('utf-8', 'surrogatepass')
    return unencrypted_bits


def eve_chooses_bases(len):
    print("AliceChooseBases")
    # Generate Alice's bases.
    eve_bases = []  # List to store Alice's bases
    for i in range(len):
        base = getrandbits(1)
        if base == 0:
            eve_bases.append("Z")
        else:
            eve_bases.append("X")
    return eve_bases


def eve_measure(eve_bases, encoded_qubits, AmountofQubit):
    print('eve measure')
    # measure Alice's qubits. Remeber that if a bit bob_bases is 0, Bob measures in the Z basis. If the bit is 1,
    eve_bits = []  # List of Bob's bits generated from the results of Bob's measurements
    for i in range(AmountofQubit):
        qc = encoded_qubits[i]
        if eve_bases[i] == "Z":  # Bob's basis is Z
            qc.measure(0, 0)  # Code to measure in the Z basis
        elif eve_bases[i] == "X":  # The case that bob_bases[i] is X
            qc.h(0)  # WRITE CODE HERE: Bob needs to add a gate here to correctly measure in the X basis. Which gate is this?
            qc.measure(0, 0)  # Measurement in the X basis
        # Now that the measurements have been added to the circuit, let's run them.
        job = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=1)
        results = job.result()
        counts = results.get_counts()
        measured_bit = max(counts, key=counts.get)
        #print(str(i) + "qubits measured")

        # Append measured bit to Bob's measured bitstring
        eve_bits.append(measured_bit)
    return eve_bits
