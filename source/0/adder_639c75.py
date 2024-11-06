# https://github.com/cpbunker/learn_qiskit/blob/08911639eb8fd94b2c5101ccfb24e6385cfc9a27/intro/adder.py
'''
https://github.com/cpbunker/learn/qiskit

https://learn.qiskit.org/course/introduction/the-atoms-of-computation
'''

import utils

import qiskit
from qiskit import QuantumCircuit;


def half_adder(n1: int, n2: int) -> int:
    '''
    use a quantum circuit to add two single bit binary integers
    '''
    assert(n1 in [0,1] and n2 in [0,1]);

    # first bit of each number is input to halfadder
    input_str = str(n1) + str(n2)
    input_circ = utils.str_to_qc(input_str);

    # half adder circuit
    halfadder = QuantumCircuit(4,2);
    halfadder.cx(0,2);
    halfadder.cx(1,2);
    halfadder.ccx(0,1,3); # toffoli
    halfadder.measure([2,3],[0,1]);

    # compose two circuits together
    qc = QuantumCircuit(len(halfadder.qubits), len(halfadder.clbits)); # zeros like
    qc.compose(input_circ, qubits = list(range(len(input_circ.qubits))), clbits = list(range(len(input_circ.clbits))), inplace = True);
    qc.compose(halfadder, qubits = list(range(len(halfadder.qubits))), clbits = list(range(len(halfadder.clbits))), inplace = True);

    return qc;

##################################################################################
# run code
if(__name__ == '__main__'):
    bits = (1,0);
    myqc = half_adder(*bits);
    print("In >> ", bits);
    print(myqc.draw());
