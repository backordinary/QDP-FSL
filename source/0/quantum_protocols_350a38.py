# https://github.com/axie66/QuantumCrypto/blob/98fc4796d24d1ad239aedf5ecd8cab4b1532058c/quantum_protocols.py
#########################################################################
# quantum_protocols.py
#
# 15-251 Project
# Quantum Cryptography Protocols in IBM Qiskit
#
# Written by Alex Xie (alexx)
#########################################################################

from qiskit import QuantumCircuit, execute, Aer
import numpy as np

#########################################################################
# B92 Implementation
# Written by me
# Inspired by BB84 implementation in Qiskit textbook:
# https://qiskit.org/textbook/ch-algorithms/quantum-key-distribution.html
#########################################################################

def encode_bit(b):
    '''Encodes bit b as a polarization state'''
    # we'll represent our photon as a qubit in a quantum circuit
    qc = QuantumCircuit(1, 1) 
    if b == 0:
        pass # 0 degree polarization
    else:
        qc.h(0) # 45 degree polarization
    qc.barrier()
    return qc

def encode_bits(bits):
    '''Encodes bits as a sequence of polarization states'''
    return [encode_bit(b) for b in bits]


def decode_states(states):
    '''Decodes a sequence of polarization states into bits'''
    backend = Aer.get_backend('qasm_simulator')
    bits = []
    successes = []
    for index, state in enumerate(states):
        choice = np.random.randint(2)
        if choice == 1:
            # measure with rectilinear filter
            state.measure(0, 0)
        else:
            # measure with diagonal filter
            state.h(0)
            state.measure(0, 0)
        result = execute(state, backend, shots=1, memory=True).result()
        out = int(result.get_memory()[0])
        if out == 1:
            # We can only get a 1 if the filter used the measure was NOT the 
            # same as the basis the photon was in.
            # If choice == 1, then the photon must have been 45 degrees, which
            # corresponds to a 1 bit.
            # If choice == 0, then the photon must have been 0 degrees, which 
            # corresponds to a 0 bit.
            bits.append(choice)
            successes.append(index)
    return bits, successes


######################
# B92 Demo
######################

print("******************************")
print("B92 Demo")
print("******************************\n")

n = 2000 # This takes a bit to run
annie_original_secret = np.random.randint(2, size=n) # Annie's original secret bits
annie_photons = encode_bits(annie_original_secret) # Annie's photons, which get sent to Britta

britta_secret, success_indices = decode_states(annie_photons)

# Annie keeps the indices that Britta was able to draw conclusions from.
annie_new_secret = [annie_original_secret[i] for i in success_indices]

assert britta_secret == annie_new_secret # Sanity check
print('Britta and Annie\'s secret keys are the same.')

# We should expect the secret key to be n/4 bits long.
# For each photon, we have a 1/2 chance of measuring with the opposite filter. 
# Then, given that we measured with the opposite filter, we have a 1/2 chance 
# of getting the basis state orthogonal to 0 or 45 degrees. Thus, we have a
# 1/2 * 1/2 = 1/4 chance of being able to determine any given photon's 
# polarization and in doing so obtain the bit Annie sent.
print('The length of the B92 secret key is:', len(britta_secret))

print('\n\n')
#########################################################################
# Kak's Protocol Implementation
# Written by me
#########################################################################

def message2bits(message):
    '''Converts message string into list of bits'''
    bits = []
    for c in message:
        ascii_bits = [int(b) for b in bin(ord(c))[2:]]
        padded_ascii_bits = (8 - len(ascii_bits)) * [0] + ascii_bits
        bits += padded_ascii_bits
    return bits

def bits2message(bits):
    '''Converts list of bits into message string'''
    result = ''
    for i in range(0, len(bits), 8):
        ascii_int = 0
        for j in range(8):
            ascii_int += bits[i + 7 - j] * 2**j
        result += chr(ascii_int)
    return result

def prepare_states(bits):
    '''Encodes bits as polarization basis states'''
    states = []
    for b in bits:
        qc = QuantumCircuit(1, 1) # photon represented as qubit
        if b == 0:
            pass # 0 degree polarization
        else:
            qc.x(0) # 90 degree polarization
        qc.barrier()
        states.append(qc)
    return states

def decode_states(states):
    '''Converts polarization basis states back to bits'''
    bits = []
    backend = Aer.get_backend('qasm_simulator')
    for state in states:
        state.measure(0, 0)
        result = execute(state, backend, shots=1, memory=True).result()
        bit = int(result.get_memory()[0])
        bits.append(bit)
    return bits

def apply_rotation_operators(states, angles):
    '''For all i, applies rotation of angle angles[i] on 
       polarization state states[i]'''
    new_states = []
    for state, angle in zip(states, angles):
        state.ry(angle, 0) # perform rotation about y-axis
        state.barrier()
        new_states.append(state)
    return new_states


######################
# Kak's Protocol Demo
######################

print("******************************")
print("Kak's Protocol Demo")
print("******************************\n")

is_adversary = 1 # Can toggle whether an adversary is trying to intercept message.
alice_message = 'I hate cilantro :(' # The message that Alice wants to send to Bob.
print('The original message:', alice_message)
alice_bits = message2bits(alice_message)

# Angles of Alice's and Bob's rotation operators (one per bit)
alice_angles = np.random.random(size=len(alice_bits)) * np.pi/2 
bob_angles = np.random.random(size=len(alice_bits)) * np.pi/2

# Stage 1: Alice encodes her bits as polarizations and applies her rotation 
# operators on all the bits.
initial_photons = prepare_states(alice_bits)
step1_photons = apply_rotation_operators(initial_photons, alice_angles)

# Stage 2: Bob applies his rotation operators on all the bits Alice sent. 
step2_photons = apply_rotation_operators(step1_photons, bob_angles)

print(f'There is {"an" if is_adversary else "no"} adversary.')
if is_adversary:
    eve_message = bits2message(decode_states(step2_photons))
    print('What Eve sees:', eve_message)

# Stage 3: Alice applies the inverses of her rotation operators on all the bits.
# Bob can now apply the inverses of his rotation operators and decode the photons.
step3_photons = apply_rotation_operators(step2_photons, -alice_angles)
bob_photons = apply_rotation_operators(step3_photons, -bob_angles)
bob_bits = decode_states(bob_photons)
bob_message = bits2message(bob_bits)
print('What Bob receives:', bob_message)