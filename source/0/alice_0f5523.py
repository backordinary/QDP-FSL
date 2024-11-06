# https://github.com/AbeerVaishnav13/Quantum-Programs/blob/73b0b8f7e93095bb44725aa60d00287a86cc61d6/QKD/Alice.py
from qiskit import QuantumCircuit

alice_key = '11100100010001001001001000001111111110100100100100010010010001010100010100100100101011111110001010100010010001001010010010110010'

alice_bases = '11000110011000100001100101110000111010011001111111110100010111010100000100011001101010100001010010101011010001011001110011111111'

def alice_prepare_qubit(qubit_index):
    ## WRITE YOUR CODE HERE
    qc = QuantumCircuit(1, 1)
    
    if alice_key[qubit_index] == '1':
        qc.x(0)
        
    if alice_bases[qubit_index] == '1':
        qc.h(0)
        
    return qc
    ## WRITE YOUR CODE HERE