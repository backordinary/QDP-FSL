# https://github.com/matiastamsi/QKD/blob/d9a2c8c7919c0de74a0b8161a3cc1ce5947d430b/main.py
# The following code follows examples from:
# https://qiskit.org/textbook/ch-algorithms/quantum-key-distribution.html
# with some own additional functionalities such as
# generating the random keys and bases with quantum circuits
# instead of using some additional random generator.

import os
from dotenv import load_dotenv
from qiskit import IBMQ, QuantumCircuit, Aer, assemble
from qiskit.visualization import plot_histogram
from IPython.display import display

try:
    provider = IBMQ.load_account()
except:
    load_dotenv()
    IBMQ.save_account(os.getenv('IBM_TOKEN'))
    provider = IBMQ.load_account()


def generate_random_bits(length):
    bits = ''
    for i in range(length):
        qc = QuantumCircuit(1,1)
        qc.h(0)
        qc.barrier()
        qc.measure(0,0)
        #display(qc.draw())
        aer_sim = Aer.get_backend('aer_simulator')
        job = aer_sim.run(assemble(qc), shots=1, memory=True)
        measured_bit = int(job.result().get_memory()[0])
        bits += str(measured_bit)
        #plot_histogram(job.result().get_counts(), filename='histogram.png')
    return bits

def encode(key, bases):
    qubits = []
    for i in range(len(key)):
        qc = QuantumCircuit(1,1)
        if bases[i] == 0:   # Z-basis
            if key[i] == 1:
                qc.x(0)
        else:               # X-basis
            if key[i] == 0:
                qc.h(0)
            else:
                qc.x(0)
                qc.h(0)
        qc.barrier()
        qubits.append(qc)
    return qubits

alice_bases = generate_random_bits(100)
bob_bases = generate_random_bits(100)
print("Alice's bases:\n" + alice_bases)
print("Bob's bases:\n" + bob_bases)


alice_key = generate_random_bits(100)
bob_key = generate_random_bits(100)
print("Alice's key:\n" + alice_key)
print("Bob's key:\n" + bob_key)

alice_qubits = encode(alice_key, alice_bases)
bob_qubits = encode(bob_key, bob_bases)

