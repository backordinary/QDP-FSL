# https://github.com/yash-srivastava19/QKD_Example/blob/e2d7674ab1c1e62a0a0d570a3ac6d2a9321db41f/qkd.py
# Quantum Key Distribution : Cryptography based on Quantum Mechanics !
""" 
Physical Implementation of Quantum Chammel : 

In classical sense, imagine a telephone line; we send electric signals through the line
that represents our message, however for a quantum channelcan be fibre optic cable through
which we can send individual photons. The polarization of the phioton can be used to 
represent a qubit.  

""" 

# The QKD protocol makes use of the fact htat measuring a qubit can change it's state.

import numpy
import matplotlib.pyplot as plt
from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from numpy.random import randint

""" 
If Alice sends a qubit in a particular state, Bob, in theory should measure the 
same state, if the message hasn't been interuppted by then.

""" 

#  qc = QuantumCircuit(1, 1)

# Alice prepares the qubit in state |+ > (0 in X-basis)

# qc.h(0)
# qc.barrier()

# Alice now sends the qubit to Bob, who measures it in X-basis

# qc.h(0)
# qc.measure(0, 0)

# print(qc.draw("text"))
# aer_sim = Aer.get_backend('aer_simulator')
# job = aer_sim.run(assemble(qc))
# plot_histogram(job.result().get_counts())
# plt.show()

# qc = QuantumCircuit(1, 1)

# Alice prepares the qubit in state |+ > (0 in X-basis)

# qc.h(0)

# Alice sends the qubits, but CArl intercepts and tries to read it

# qc.measure(0, 0)
# qc.barrier()

# Carl now sends the qubit to Bob, who measures it in X-basis

# qc.h(0)
# qc.measure(0, 0)

# print(qc.draw("text"))
# aer_sim = Aer.get_backend('aer_simulator')
# job = aer_sim.run(assemble(qc))
# plot_histogram(job.result().get_counts())
# plt.show()

# We can see there that Bob has now a 50% chance of measuring 1 or 0, so Alice will 
# know something is wrong with their quantum channel !!


""" 
QKD Protocol involves repeating this process enough times so that an eavesdropper
has a negligible chance of getting away with this interception. Steps Involved :

Step 1 : 

--> Alice chooses a string of random bits, and a random choice of basis for each bit.
Alice keeps these two pieces of information to herself. 

Step 2:

--> Alice then encodes each bit into a string of qubits using the basis she chose, and
sends to Bob.

Step 3:

--> Bob measures each qubit at random, using a random basis, and keeps the result private

Step 4:

--> Bob and Alice then publicly share which basis they used for each qubit. If Bob
measured a qubit in the same basis as Alice prepared it, they use this to form part 
of their shared secret key, otherwise disccard the information for that bit.

Step 5:

--> Finally, Bob and Alice share a random sample of their keys, and if the samples 
match, they can be sure(a small margin of error is allowed) that their transmission 
is successful.

"""

# E2E Example - Without Interception

numpy.random.seed(0)
N = 100


# Creates a list of QuantumCircuits, each representing a single qubit in Alice's message
def encode_message(bits, bases):
    message = []

    for i in range(N):
        qc = QuantumCircuit(1, 1)
        if bases[i] == 0:   # Prepare qubit in Z-basis ; 0 encoded
            if bits[i] == 0:
                pass
            else: 
                qc.x(0)
        else:              # Prepare qubit in X-basis ; 1 encoded
            if bits[i] == 0:
                qc.h(0)
            else:
                qc.x(0)
                qc.h(0)
        qc.barrier()
        message.append(qc)
    return message    


def measure_message(message, bases):
    # backend = Aer.get_backend('aer_simulator')
    measurements = []

    for q in range(N):
        if bases[q] == 0:            # Measuring in Z-basis
            message[q].measure(0, 0)
        
        if bases[q] == 1:            # Measuring in X-basis 
            message[q].h(0)
            message[q].measure(0, 0)

        aer_sim = Aer.get_backend('aer_simulator')
        qobj = assemble(message[q], shots=1, memory=True)
        result = aer_sim.run(qobj).result()
        measured_bit = int(result.get_memory()[0])
        measurements.append(measured_bit) 
    
    return measurements

def remove_garbage(a_bases, b_bases, bits):
    good_bits = []
    for q in range(N):
        if a_bases[q] == b_bases[q]:
            good_bits.append(bits[q])
    
    return good_bits

def sample_bits(bits, selection):
    sample = []
    for i in selection:
        i = numpy.mod(i, len(bits))
        sample.append(bits.pop(i))
    
    return sample


alice_bits = randint(2, size=N) 

# Bases : Which qubits are encoded in which bases..
alice_bases = randint(2, size=N)
message = encode_message(alice_bits, alice_bases)


# Bob's bases for decrypting the message.
bob_bases = randint(2, size=N)
bob_result = measure_message(message, bob_bases)


# Step 4.
alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
bob_key = remove_garbage(alice_bases, bob_bases, bob_result)
print(alice_key)
print(bob_key)

# Finally, Bob and Alice compare a random selection of the bits in their keys to
# make sure the protocol has worked correctly

SS = 15     # Sample size

bit_selection = randint(N, size=SS)

bob_sample = sample_bits(bob_key, bit_selection)
print(f'Bob Sample = {bob_sample}')

alice_sample = sample_bits(alice_key, bit_selection)
print(f'Bob Sample = {alice_sample}')


print(alice_sample == bob_sample)

