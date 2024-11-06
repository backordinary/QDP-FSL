# https://github.com/anshsingal/CIRQuIT-QC-Internship/blob/c35c9196b68b891ecc40d2b65fad7b3247b817ad/QKD_with_eve/Alice2.py
from qiskit import QuantumCircuit
import random

random.seed(84) # DO NOT CHNAGE THIS SEED VALUE

alice_key = ''

alice_bases = ''

# alice_key = '0111001'
# alice_bases = '1101000'

def alice_prepare_qubit(num_qubits, alice_key = None):
#     print(str((random.getrandbits(num_qubits))))
#     print(str((random.getrandbits(num_qubits))))
    alice_bases = str('{:050b}'.format(random.getrandbits(num_qubits)))
    if alice_key == None:
        alice_key = str('{:050b}'.format(random.getrandbits(num_qubits)))
#     print(str((random.getrandbits(num_qubits))))
#     print(str((random.getrandbits(num_qubits))))
    alice_qubit_circuits = []
#     print(alice_bases)
#     print(alice_key)
    for i in range(len(alice_bases)):
        alice_qubit_circuits.append(generate_circuit(alice_key[i]+alice_bases[i]))
    return alice_qubit_circuits, alice_bases, alice_key
    
def generate_circuit(code):
    qubit = QuantumCircuit(1,1)
    if code == '00':
        qubit.i(0)
    elif code == '01':
        qubit.h(0)
    elif code == '10':
        qubit.x(0)
    elif code == '11':
        qubit.x(0)
        qubit.h(0)
    return qubit