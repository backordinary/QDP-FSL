# https://github.com/lfd/arcs2022/blob/9535c4cda5d07fc80ca0b17df7ee927c0f9deb64/src/qaoa.py
# coding: utf-8
# Copyright (c) Siemens AG, 2022
# SPDX-License-Identifier: GPL-2.0


import numpy as np
import itertools
from qiskit import QuantumCircuit


def get_simple_circ_qaoa_from_qubo(Q, theta):

    """
    Input: 
    Q: Qubo-Matrix for problem with constraints being built in as Hamiltonian penalty terms
    theta: List of parameters for unitaries: [gamma1,... ,gammap, beta1,... ,betap] for p layers of QAOA
    offset: Optional offset of qubo matrix, can also be set=0
    
    Output: Qiskit QuantumCircuit object for QAOA circuit containing p=len(theta)/2 layers of Problem and Mixing unitaries
    Uses the calculated coefficients
    """
    
    # get number of qubits
    num_qubits = len(Q) 

    # index tuples for 2-qubit gates: All possible combinations [i,j] for i!=j and i,j \in {1,num_qubits} whereas [i,j] = [j,i]
    tq_id = list(itertools.combinations(np.linspace(0, num_qubits-1, num_qubits, dtype=int), 2))
    
    
    """
    Convert Hamiltonian directly to Ising model to calculate the coefficients of the Pauli-Terms. 
    Contains only linar and quadratic terms (2D qubo-matrix), resulting in single- and two-qubit rotation gates. 
    Ising Hamiltonian: H = \Sum_i h_i * s_i + \Sum_{i<j} J_{ij} s_i s_j + C
    Qubo Hamiltonian: H = x^T Q x + offset
    """
    
    
    # Linear ising terms: Combinations with one Pauli-Z and I otherwise
    Ising_h = []
    for i in range(num_qubits):
        Ising_h.append(0.5*Q[i,i] + 0.25*np.sum([(Q[i,j]+Q[j,i]) for j in range(num_qubits) if j != i]))
     
    
    # quadratic Ising terms: Use J[i,j] only for unequal indices [[i,j]
    Ising_J = 0.25*Q
    
    #offset term, if needed
    #Ising_C = np.sum(np.diag(Q))*0.5 + np.sum(list(Q[tq_id[j]] for j in range(len(tq_id))))*0.25 + offset
    

    # dictionaries: {index, the indices of the qubit(s) to which Z-operators are applied (all other get identities), Ising-coefficient}
    sq_gates={}
    tq_gates={}
    for j in range(num_qubits):
        sq_gates[j] = j, Ising_h[j]
    for j in range(len(tq_id)):
        tq_gates[j] = tq_id[j], Ising_J[tq_id[j][0], tq_id[j][1]] 



    """
    Construct different layers of Hamiltonian which are alternated.   
    """
    
    p = len(theta)//2  # number of alternating unitaries
    beta = theta[:p]
    gamma = theta[p:]       
    
    qc = QuantumCircuit(num_qubits)
    
    # Intial state: Eigenstate of mixing Hamiltonian
    for i in range(0, num_qubits):
        qc.h(i)
    
    # QAOA layers
    for irep in range(0, p):
        

        # Problem Hamiltonian 
        for key in tq_gates.keys():  # two-qubit gates
            qc.rzz(2 * gamma[irep] * tq_gates[key][1], tq_gates[key][0][0], tq_gates[key][0][1])
            
        for key in sq_gates.keys():  # single-qubit gates
            qc.rz(2 * gamma[irep] * sq_gates[key][1] , sq_gates[key][0]) 

        # Mixing Hamiltonian
        for i in range(0, num_qubits):
            qc.rx(2 * beta[irep], i)
            

    
    # important for execution. Adds the apropriate number of classical bits automatically to store the measurement outcome
    qc.measure_all()
        
    return qc