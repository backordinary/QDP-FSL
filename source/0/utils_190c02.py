# https://github.com/cpbunker/learn_qiskit/blob/08911639eb8fd94b2c5101ccfb24e6385cfc9a27/Spring2022QuantumChallenge/utils.py
'''
https://github.com/cpbunker/learn/qiskit
'''

import numpy as np

# qiskit
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# backend
from qiskit.providers.aer import AerSimulator


def basis_strings(n: int) -> list:
    '''
    given a system of n qubits, output list of all the bit strings forming the
    computational basis

    e.g. n=2 returns ['00','01','10',11']
    '''
    assert(isinstance(n, int));

    # logical basis in decimal
    b_ints = np.array(range(2**n)); 

    # convert to bit strings
    b_strings = np.full(np.shape(b_ints), '0'*n);
    for i in range(len(b_strings)):
        bit = bin(b_ints[i])[2:];
        bit = '0'*(n-len(bit)) + bit; # standardize length
        b_strings[i] = bit;

    return list(b_strings);   

def output(qc: QuantumCircuit) -> None:
    '''
    Outputs helpful information about the quantum circuit
    '''
    assert(isinstance(qc, QuantumCircuit));

    # display the circuit
    print(qc.draw(output = "text"));

    # simulate the circuit
    if(len(qc.clbits)):
        job = AerSimulator().run(qc); # physically runs the qc
        jobd = job.result(); # results of simulation stored in dictionary
        print("Out >> ",jobd.get_counts());


def basis_op(qc: QuantumCircuit) -> None:
    '''
    Given a quantum circuit acting on n qubits, operates on the 2**n basis states
    More general version of qiskit.quantum_info.Operator(QuantumCircuit)
    '''
    assert(isinstance(qc, QuantumCircuit));
    print(qc.name+" operation:");

    # iter thru basis in decimal
    for i in range(2**len(qc.qubits)):

        state = Statevector.from_int(i, 2**len(qc.qubits));
        print("\n - ",state,"\n    ->",state.evolve(qc));
        
                
