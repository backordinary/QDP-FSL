# https://github.com/cpbunker/learn_qiskit/blob/08911639eb8fd94b2c5101ccfb24e6385cfc9a27/intro/grover.py
'''
https://github.com/cpbunker/learn/qiskit

https://learn.qiskit.org/course/introduction/grovers-search-algorithm
'''

import utils

import qiskit
from qiskit import QuantumCircuit;
from qiskit.quantum_info import Statevector

##################################################################################
# oracles

def oracle1(soln: str) -> QuantumCircuit:
    '''
    the clbit string "soln" is the only solution to the sat problem. this function constructs a unitary
    oracle which applies a phase (-1) to the soln and leaves all other same-length input bits unchanged
    '''
    for c in soln: assert(c in ['0','1']);
    assert(len(soln) == 2); # two bit case for now

    # rotate the soln to the 11 state
    qc = QuantumCircuit(len(soln));
    for ci in range(len(soln)):
        if(soln[ci] == '0'):
            qc.x(len(soln)-1-ci);

    # perform cz to phase the soln
    qc.cz(0,1);

    # rotate back
    for ci in range(len(soln)):
        if(soln[ci] == '0'):
            qc.x(len(soln)-1-ci);

    return qc;


##################################################################################
# other useful circuits

def superimposer(nqubits: int) -> QuantumCircuit:
    '''
    circuit that preps the superposition state
    '''

    super = QuantumCircuit(nqubits);
    for qi in range(nqubits):
        super.h(qi);
    return super;


def diffuser(nqubits: int) -> QuantumCircuit:
    '''
    circuit that reflects the state through the axis perpendicular to an equal superposition
    of all logical basis states
    '''
    assert(isinstance(nqubits, int));
    assert(nqubits == 2); # for now

    # do the diffuser by transforming superposition to 11, doing cz to reflect perp to 11, and transforming back
    diff = QuantumCircuit(nqubits);
    for qi in range(nqubits): 
        diff.h(qi); # takes back to 0
        diff.x(qi); # takes to 1

    # reflect perp to 11
    diff.cz(0,1);

    # transform back with inverse circuit
    for qi in range(nqubits): 
        diff.x(qi); # x is own inverse
        diff.h(qi); # h is own inverse

    return diff;


##################################################################################
# run code

if(__name__ == '__main__'):
    
    #### perform grovers algorithm for 2 qubit system !
    nqubits = 2;
    sol = '11'
    print("In >> ",sol);
    
    # make a grover circuit out of the superimposer, oracle and diffuser
    grover = QuantumCircuit(nqubits);
    grover = grover.compose(superimposer(nqubits));
    grover = grover.compose(oracle1(sol));
    grover = grover.compose(diffuser(nqubits));
    grover.measure_all();

    # see results
    utils.output(grover);
       

