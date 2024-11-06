# https://github.com/dorkamotorka/QuantumKeyExchange/blob/69b87eac0c31c3d80bc6f8655c2a9bf702966547/exchange.py
#!/usr/bin/env python

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from numpy.random import randint
import numpy as np

def encode_message(bits, bases):
    message = []
    for i in range(n):
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

def measure_message(message, bases):
    backend = Aer.get_backend('aer_simulator')
    measurements = []
    for q in range(n):
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

def sample_bits(bits, selection):
    sample = []
    for i in selection:
        # use np.mod to make sure the
        # bit we sample is always in
        # the list range
        i = np.mod(i, len(bits))
        # pop(i) removes the element of the
        # list at index 'i'
        sample.append(bits.pop(i))
    return sample

def remove_garbage(a_bases, b_bases, bits):
    good_bits = []
    for q in range(n):
        if a_bases[q] == b_bases[q]:
            # If both used the same basis, add
            # this to the list of 'good' bits
            good_bits.append(bits[q])
    return good_bits

if __name__ == '__main__':
    n = 100
    np.random.seed(seed=0)

    # Step 1: Alice generates random bits and random bases
    alice_bits = randint(2, size=n)
    alice_bases = randint(2, size=n)
    message = encode_message(alice_bits, alice_bases)

    # Step 2(optional): Eve eavesdroppes Alices message and trasmits it to Bob
    '''
    eve_bases = randint(2, size=n)
    intercepted_message = measure_message(message, eve_bases)
    '''

    ## Step 3: Bob decides in which basis to measure in:
    bob_bases = randint(2, size=n)
    bob_results = measure_message(message, bob_bases)

    ## Step 4: Communicate through public channel which bases Bob guessed correctly
    alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
    bob_key = remove_garbage(alice_bases, bob_bases, bob_results)

    ## Step 5: Check if selected bits of keys match 
    sample_size = 15
    bit_selection = randint(n, size=sample_size)
    bob_sample = sample_bits(bob_key, bit_selection)
    alice_sample = sample_bits(alice_key, bit_selection)
    if not (alice_sample == bob_sample):
        print("Attacker intercepted the message, aborting!")
        exit()

    print("Established key is \n %s" % alice_key)
