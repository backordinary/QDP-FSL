# https://github.com/immoth/Hubbard_symmetries/blob/a78622d48dc9e2645841db423f2f9a21b419322c/Entropy/Object_Oriented/Define_Pauli_Measure.py


from qiskit import Aer
from qiskit.visualization import *
from qiskit import quantum_info as qi
import copy
from Define_Translator import translate, measure_rho
import numpy as np
from Define_Paulis import Mdot

def measure_pauli(p_label,psi0,method = 'simple'):
    
    #Simple method of calculation####
    if method == 'simple':
        rho = translate(psi0)
        Op = qi.Operator.from_label(p_label)
        return np.trace( Mdot([Op,rho]) )
    #################################
        
    #apply rotations#################
    psi = copy.deepcopy(psi0)
    pauli_qs = []
    Z_label = ''
    Q = len(p_label)
    for q,p in enumerate(p_label):
        if p == 'X':
            psi.ry(-np.pi/2,Q-1-q)
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'Y':
            psi.rx(np.pi/2,Q-1-q)
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'Z':
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'I':
            pauli_qs.append(0)
            Z_label += 'I'
    ################################
    
    #Using matrix multiplication####
    if method == 'matrix':
        # add results
        rho = translate(psi)
        r = measure_rho(rho)
        z_measure = 0
        for key in list(r.keys()):
            n = 0
            for q in range(len(key)):
                if key[q] == '1' and pauli_qs[q] == 1:
                    n += 1
            z_measure += (-1)**n * r[key] 
        return z_measure
    #################################
    
    #Using the qasm simulator########
    if method == "simulator":
        sim = Aer.get_backend("qasm_simulator")
        psi.measure(psi.qubits,psi.clbits)
        r = execute(psi, backend = sim,shots = 10000).result().get_counts()
        z_measure = 0
        total = 0
        for key in list(r.keys()):
            n = 0
            for q in range(len(key)):
                if key[q] == '1' and pauli_qs[q] == 1:
                    n += 1
            z_measure += (-1)**n * r[key] 
            total += r[key]
        return z_measure/total
    ###################################
    
    raise NameError(method + ' is not a recognized method')
    return method + ' is not a recognized method'

def measure_E(H_paulis,cir,method = 'simple'):
    paulis = list(H_paulis.keys())
    e = 0
    for p in paulis:
        #print(p, H_paulis[p] , measure_pauli(p,cir,method))
        ep = H_paulis[p] * measure_pauli(p,cir,method)
        e += ep
    return e

def apply_rotations(p_label,psi0):
    psi = copy.deepcopy(psi0)
    pauli_qs = []
    Z_label = ''
    Q = len(p_label)
    for q,p in enumerate(p_label):
        if p == 'X':
            psi.ry(-np.pi/2,Q-1-q)
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'Y':
            psi.rx(np.pi/2,Q-1-q)
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'Z':
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'I':
            pauli_qs.append(0)
            Z_label += 'I'
    psi.measure(psi.qubits,psi.clbits)
    return psi,pauli_qs
    
def collect_circuits(H_paulis,cir_in):
    circs = []
    labels = []
    paulis = list(H_paulis.keys())
    for p in paulis:
        circ,label = apply_rotations(p,cir_in)
        circs.append(circ)
        labels.append([H_paulis[p],label])
    return circs,labels
