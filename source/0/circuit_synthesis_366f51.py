# https://github.com/dclarino/ngc-dd/blob/00aa34e9b275b046a1cb724759d96793de3f3bca/src/circuit_synthesis.py
from qiskit import QuantumCircuit,QuantumRegister
from dd.autoref import BDD

#from src.circuit_to_logic import *
import logging
import os
import sys
sys.path.append(os.path.join('../src'))
# print(sys.path)

from circuit_to_logic import *

def calculate(qc,bdd):
    
    qubit_states = []
    for i in range(len(qc.qubits)):                                      # generating qubit states
        qubit_states.append('q' + str(i))
    
    qc_gates = retrieve_gates(qc)                                        # generating gates list
    
    total_phase = ['True', 'False', 'False', 'False']                    # initializing total_phase, with total_phase[0] indicating that 0 phase has to be added
    for gate in qc_gates:
        print(gate)
        if gate[0] == 'cx':                                              # applying the individual gates
            qubit_states[gate[1][1]] = cnot_string(
                qubit_states[gate[1][0]], qubit_states[gate[1][1]])
        elif gate[0] == 'x':
            qubit_states[gate[1][0]] = not_string(qubit_states[gate[1][0]])
        elif gate[0] == 'ccx':   
            print(gate[1][2])                                        # phase change applies only for the toffoli gate
            if gate[1][2] == 3:                                          # if the phase change is in the ancilla bit, we need to calculate it 
                add_phase = ['True', 'True', 'True', 'True']             # initializing add_phase
                                                                       # applying and gates to check which add_phase becomes true
                add_phase[1] = and_string(qubit_states[gate[1][0]], and_string(qubit_states[gate[1][1]], not_string(qubit_states[gate[1][2]])))
                add_phase[2] = and_string(qubit_states[gate[1][0]], and_string(qubit_states[gate[1][2]], not_string(qubit_states[gate[1][1]])))
                add_phase[3] = and_string(qubit_states[gate[1][0]], and_string(qubit_states[gate[1][1]], qubit_states[gate[1][2]]))
                                                                         # if none of the above three varibles are true, then add_phase[0] is set to true as no phase is to be added
                add_phase[0] = not_string(or_string(add_phase[1], or_string(add_phase[2], add_phase[3])))
                print(add_phase)
                print(total_phase)
                temp_phase = [' ', ' ', ' ', ' ']
                for index in range(0, 4):                                 
                    string_index = total_phase[index]                                # if add_phase[k] and total_phase[j] is true, it means that current phase is j * pi/2 and k * pi / 2 phase has to be added
                                                                         # new_phase = (j + k) * (pi /2) modulo 2*pi  => new_phase = ((j + k) mod 4 ) * pi/2
                                                                         # the subsequent loop ensures that the (j + k) % 4 index is set to true when there is any such pair (j, k) such that add_phase[k] and total_phase[j] is true
                    for j in range(0, 4):
                        for k in range(0, 4):
                            if (j + k) % 4 == index:
                                string_index = or_string(string_index, and_string(total_phase[j], add_phase[k]))
                    
                    temp_phase[index] = string_index
                total_phase = temp_phase
            qubit_states[gate[1][2]] = toffoli_string(qubit_states[gate[1][0]], qubit_states[gate[1][1]], qubit_states[gate[1][2]])
    print(total_phase)
    bdd_exprs = [bdd.add_expr(total_phase[i]) for i in range(0, 4)]
    return bdd_exprs

def synthesize(phase_function,bdd,qr):
    return [QuantumCircuit(qr),QuantumCircuit(qr), QuantumCircuit(qr),QuantumCircuit(qr)]


