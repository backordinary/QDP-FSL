# https://github.com/cpbunker/learn_qiskit/blob/08911639eb8fd94b2c5101ccfb24e6385cfc9a27/Spring2022QuantumChallenge/mbl.py
'''
IBM Quantum Spring Challenge 2022
Exercise 2: Quantum Random Walks and Localization 
'''

import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit, QuantumRegister
import qiskit.quantum_info as qi

def get_imbalance(state):
    ###EDIT CODE BELOW
    ### HINT: MAKE SURE TO SKIP CALCULATING IMBALANCE OF THE |00...0> STATE
    imbalance_val=np.complex128(0);
    n_qubits = len(state.dims());
    assert state.is_valid();

    # decompose state into basis states
    statedict = state.to_dict();
    for basis_key in statedict:
        basis_imbalance = np.complex128(0); # to be determined
        
        # special cases
        if('1' not in basis_key): # ie if the null state
            basis_imbalance += 0;
        else:
            # just count the bit composition
            Ne, No = np.complex128(0), np.complex128(0);
            for qubiti in range(n_qubits):
                if(qubiti % 2 == 0): # even site
                    Ne += int(basis_key[qubiti]);
                    #print(basis_key,qubiti,Ne);
                else: # odd site
                    No += int(basis_key[qubiti]);
                    
            basis_imbalance = (Ne - No)/(Ne + No);
            
        # add in weighted imbalance
        basis_coef = statedict[basis_key]
        imbalance_val += basis_coef*np.conj(basis_coef)*basis_imbalance;
        
        # debug
        #print('>> ', basis_key, basis_coef*basis_coef, basis_imbalance)
         
    imbalance_val = np.real(imbalance_val); 
    ###DO NOT EDIT BELOW
    
    return imbalance_val


#################################################################
#### test code
if(__name__ == '__main__'):
    print(np.__version__)
    import sys
    print(sys.version_info)
    xs = []
    xs.append(qi.Statevector.from_label('1'));
    xs.append(qi.Statevector.from_label('0111'));
    xs.append(np.sqrt(2/3)*qi.Statevector.from_label('0111') + np.sqrt(1/3)*qi.Statevector.from_label('1011') );
    xs.append(qi.Statevector.from_label('0000')/np.sqrt(2) + qi.Statevector.from_label('0111')/np.sqrt(2) );
    xs.append(qi.Statevector.from_label('0000')/np.sqrt(9/8) + qi.Statevector.from_label('1011')/np.sqrt(9/1) );
    for x in xs:
        assert(x.is_valid() ); 
        print(get_imbalance(x));

