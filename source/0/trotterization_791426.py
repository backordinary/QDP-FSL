# https://github.com/brower/QuLAT/blob/da7911ac972ddb66d282a0ff14fd30eee595fade/Hiroki/quantum_circuit/trotterization.py
'''
Created on Sep 26, 2019

@author: kwibu
'''
import numpy as np
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from utils.circuits import add_crx, add_ccrx
from qiskit.quantum_info.operators import Operator


def trotter_pauli(q_circuit, qr, pauli_op, deltaT, unitary_sim = False):
    """
    Trotter step for one general pauli term
    Arguments:
    q_circuit: qiskit.QuantumCircuit, the target circuit
    qr: qiskit.QunatumRegister, the input qubit state
    pauli_op: operators.pauli_hamiltonian.PauliOperator, the target pauli operator
    deltaT: small time step
    unitary_sim: bool, True if the simulation is to get the unitary. Due to the bug for qiskit Aer >= 0.3.0.
    
    Return:
    qiskit.QuantumCircuit, the circuit after added the trotterization step.
    """ 
    target_indices = list(sorted(pauli_op.pauli.keys()))
    # Add a H or Rx(pi/2) gate to qubits corresponding to X or Y operators, respectively. 
    for j in target_indices:
        if pauli_op.pauli[j] == "X":
            q_circuit.h(qr[j])
        elif pauli_op.pauli[j] == "Y":
            q_circuit.rx(np.pi/2, qr[j])
    
    # Add cnot gates to target qubits
    for i in range(len(target_indices)-1):
        q_circuit.cx(qr[target_indices[i]], qr[target_indices[i+1]])
    # Add e^(-i*coef*dt*Z) to the last qubit
    q_circuit.u1(pauli_op.coef*deltaT, qr[target_indices[-1]])
    if unitary_sim:
        q_circuit.unitary(Operator([[0, 1], [1, 0]]), [target_indices[-1]])
    else:
        q_circuit.u3(np.pi, 0, np.pi, qr[target_indices[-1]])
    q_circuit.u1(-pauli_op.coef*deltaT, qr[target_indices[-1]])
    if unitary_sim:
        q_circuit.unitary(Operator([[0, 1], [1, 0]]), [target_indices[-1]])
    else:
        q_circuit.u3(np.pi, 0, np.pi, qr[target_indices[-1]])
    
    #q_circuit.unitary(Operator([[np.exp(-1j*pauli_op.coef*deltaT), 0], [0, np.exp(1j*pauli_op.coef*deltaT)]]), [target_indices[-1]], label='rzz')
    
    # Add cnot gates to target qubits
    for i in range(len(target_indices)-1)[::-1]:
        q_circuit.cx(qr[target_indices[i]], qr[target_indices[i+1]])
    
    # Uncompute
    for j in target_indices:
        if pauli_op.pauli[j] == "X":
            q_circuit.h(qr[j])
        elif pauli_op.pauli[j] == "Y":
            q_circuit.rx(-np.pi/2, qr[j])
    return q_circuit

def trotter_electric(q_circuit, target_indices, coef, deltaT, unitary_sim = False, further_opt = False):
    """
    Trotter step for one electric term
    (S^z + S^z)^2
    Arguments:
    q_circuit: qiskit.QuantumCircuit, the target circuit
    qr: qiskit.QunatumRegister, the input qubit state
    target_inidices: list of int, the target indices of the qubits
    coef: float, coefficient corresponding to the term
    deltaT: small time step
    unitary_sim: bool, True if the simulation is to get the unitary. Due to the bug for qiskit Aer >= 0.3.0.
    further_opt: bool, if True, then the CNOTs in the uncomputing stage are combined with those in trotter_coupling and removed. 
    
    Return:
    qiskit.QuantumCircuit, the circuit after added the trotterization step.
    """ 
    assert(len(target_indices)==2), "List of target indices for a coupling term with invalid length is given."
    q_circuit.cx(target_indices[0], target_indices[1])
    # Add e^(-i*coef*dt*Z) to the last qubit

    q_circuit.u1(coef*deltaT, target_indices[1])
    if unitary_sim:
        q_circuit.unitary(Operator([[0, 1], [1, 0]]), [target_indices[-1]])
    else:
        q_circuit.u3(np.pi, 0, np.pi, target_indices[-1])
    q_circuit.u1(-coef*deltaT, target_indices[-1])
    if unitary_sim:
        q_circuit.unitary(Operator([[0, 1], [1, 0]]), [target_indices[-1]])
    else:
        q_circuit.u3(np.pi, 0, np.pi, target_indices[-1])
    
    if not further_opt:
        q_circuit.cx(target_indices[0], target_indices[1])
    return q_circuit

def trotter_coupling(q_circuit, target_indices, coef, deltaT, further_opt = False):
    """
    Trotter step for one coupling term
    S^+ otimes S^- + h.c.
    Arguments:
    q_circuit: qiskit.QuantumCircuit, the target circuit
    qr: qiskit.QunatumRegister, the input qubit state
    target_inidices: list of int, the target indices of the plaquette
    coef: float, coefficient corresponding to the plaquette term
    deltaT: small time step
    further_opt: bool, if True, then the CNOTs in the incomputing stage are combined with those in trotter_electric and removed. 
    
    Return:
    qiskit.QuantumCircuit, the circuit after added the trotterization step.
    """ 
    assert(len(target_indices)==2), "List of target indices for a coupling term with invalid length is given."
    if not further_opt:
        q_circuit.cx(target_indices[1], target_indices[0])
    q_circuit = add_crx(q_circuit, coef*deltaT, target_indices[0], target_indices[1])
    q_circuit.cx(target_indices[1], target_indices[0])
    return q_circuit

def trotter_plaquette(q_circuit, target_indices, coef, deltaT):
    """
    Trotter step for one plaquette term
    S^+ otimes S^+ otimes S^+ + h.c.
    Arguments:
    q_circuit: qiskit.QuantumCircuit, the target circuit
    qr: qiskit.QunatumRegister, the input qubit state
    target_inidices: list of int, the target indices of the plaquette
    coef: float, coefficient corresponding to the plaquette term
    deltaT: small time step
    Return:
    qiskit.QuantumCircuit, the circuit after added the trotterization step.
    """ 
    assert(len(target_indices)==3), "List of target indices for a plaquette term with invalid length is given."
    q_circuit.cx(target_indices[0+1], target_indices[0])
    q_circuit.u3(np.pi, 0, np.pi, target_indices[0])    
    q_circuit.cx(target_indices[1+1], target_indices[1])
    q_circuit.u3(np.pi, 0, np.pi, target_indices[1])    
    # Add CCRX gate
    q_circuit = add_ccrx(q_circuit, coef*deltaT, target_indices[0], target_indices[1], target_indices[2])
    
    # Add cnot gates to target qubits
    for i in range(2)[::-1]:
        q_circuit.u3(np.pi, 0, np.pi, target_indices[i])
        #q_circuit.unitary(Operator([[0, 1], [1, 0]]), [target_indices[i]])
        q_circuit.cx(target_indices[i+1], target_indices[i])
    
    return q_circuit

    
    
    