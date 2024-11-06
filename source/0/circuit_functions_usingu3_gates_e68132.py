# https://github.com/kaykhos/clifford_project/blob/574255e8e72d4b718ee18d5af8d3c6fc7656d3a5/week2/circuit_functions_usingU3_gates.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:54:26 2020

@author: kiran

Some code copied from qcoptim to generate circuits

TODO: How does U3 impliment total phases?
"""

#%% Imports
import qiskit as qk
import numpy as np


# Need in this name space for qiskits eval() of parameters
pi = np.pi



#%% Function deffinitions
def newCircuit(nb_qubits = 4, 
               depth = 1,
               verbose = False):
    """
    Creates a new random 'cliffod' circuit"""
    
    def entangle_layer(circ):
        """
        Creates a linear entangeling layer"""
        for ii in range(0,circ.num_qubits-1, 2):
            circ.cx(ii,ii+1)
        for ii in range(1,circ.num_qubits-1, 2):
            circ.cx(ii,ii+1)
    
    def rotaiton_layer(circ):
        """
        Creates a layer of single qubit rotations based on the list 'single_rotatoins'"""
        cliffod_params = _newCliffordParams(circ.num_qubits)
        for ii, param in enumerate(cliffod_params):
            circ.u(*param, ii)
    
    # Construct circuit
    circuit = qk.QuantumCircuit(nb_qubits)

    # Apply first rotation layer (else CX layer does nothing)
    rotaiton_layer(circuit)
    
    # Loop though and alternate rotation and entangelment layers
    for ii in range(depth):
        entangle_layer(circuit)
        circuit.barrier() # this just makes circ.draw() look better
        rotaiton_layer(circuit)
    if verbose:
        print(circuit)
    return circuit


def updateCircuit(circuit,
                  verbose = False):
    """
    Takes an input circuit and switches exactly 1 single qubit gate to something 
    else. (Only tested with circuis made with newCircuit())
    """
 
    # Convert circuit to qasm string so we can use string processing to switch
    qasm = circuit.qasm().split(';')
    
    
    # Make sure the gate you choose is not a cx gate
    gate_to_switch = np.random.randint(3,len(qasm)-1)
    while qasm[gate_to_switch][1:3] == 'cx' or qasm[gate_to_switch][1:3] == 'ba':
        gate_to_switch = np.random.randint(3,len(qasm)-1)
    
    # Get a new gate and make sure it's different form the current gate
    this_gate = qasm[gate_to_switch][2:].split(' ')[0]
    new_gate = _newCliffordParams()
    while new_gate == eval(this_gate):
        new_gate = _newCliffordParams()
        
    # Update the new parameters in the qasm string
    qasm[gate_to_switch] = '\nu' + str(new_gate) + ' ' + qasm[gate_to_switch].split(' ')[1]
    
    qasm = ';'.join(qasm)    
    circuit = qk.QuantumCircuit.from_qasm_str(qasm)
    
    if verbose:
        print(circuit)
        
    return circuit


def _u3ParamsFromGateMatrix(gate):
    """
    Takes in input single qubit gate as a matrix, and outputs the 
    required qiskit u3 params"""
    circ = qk.QuantumCircuit(1)
    circ.unitary(gate, 0)
    circ = circ.decompose()
    params = circ.qasm().split('u3')[1].split(' ')[0]
    return eval(params)


def _newCliffordParams(count=1):
    """
    Uses qiskits inbuilt single qubit clifford gate sampler and returns the 
    u3 parameters required. count = nubmer of random clifford circuits you want
    """
    gates = [qk.quantum_info.random_clifford(1).to_matrix() for ii in range(count)]
    params = [_u3ParamsFromGateMatrix(gg) for gg in gates]
    if count == 1:
        return params[0]
    else:
        return params
        

#%% Quick test to make sure everyhing goes smoothly
if __name__ == '__main__':
    
    # Create circuit
    circuit = newCircuit(4, 4, True)
    
    # Update it 50 times and check for errors
    for ii in range(50):
        circuit = updateCircuit(circuit)
    
    print('passed basic test')