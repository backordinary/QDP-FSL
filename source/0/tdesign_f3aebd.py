# https://github.com/BrudeSim/tDesign/blob/deac6a944332ecb1d51ed4c2495a5a5f787d3381/tDesign.py
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:14:34 2019

@author: Bradley Pearlman
"""
from qiskit import *
from qiskit import QuantumRegister
from qiskit import Aer
from qiskitQuantumInfo import *
import networkx as nx
import random
from michaelCuts618 import *
from qiskit.providers.aer import QasmSimulator
from math import pi

backend = Aer.get_backend('qasm_simulator')

simulator = Aer.get_backend('statevector_simulator')

###############################################################################

# This script is used to demonstrate the simulation of a large Haar random unitary
# circuit from a composition of Haar random two-qubit gates connected in the
# standard even-odd alternation.  Additionally, it demonstrates a clustered-circuit
# simulation which approximates the Haar distribution worse but is ammenable to cutting.

###############################################################################

# This generates connectivity for the alternating laters in the even-odd paradigm

def graph_layers(n):
    if (n % 2) != 0:
        return 'Please Choose an even number';
    else:
        l_0 = nx.Graph()
        l_1 = nx.Graph()
        for i in range(n):
            l_0.add_node(i)
            l_1.add_node(i)
        for i in range(n):
            if (i % 2) == 0:
                l_0.add_edge(i,i+1)
        for i in range(n-1):
            if (i % 2) == 1:
                l_1.add_edge(i, i+1)
        l_1.add_edge(0,n-1)
        pos_0 = {}
        pos_1 = {}
        for i in range(n):
            pos_0.update({i : (math.cos(2*pi*(i+0.5)/n), math.sin(2*pi*(i+0.5)/n))})
            pos_1.update({i : (2.5 + math.cos(2*pi*(i+0.5)/n), math.sin(2*pi*(i+0.5)/n))})
        return l_0, l_1, pos_0, pos_1;
    
l_0, l_1, pos_0, pos_1 = graph_layers(8)

###############################################################################

# Same as graph_layers except we will remove some edges to make cutting possible
def cutable_graph_layers(n):
     l_0, l_1, pos_0, pos_1 = graph_layers(n)
     m_0 = l_0.copy()
     m_0.remove_edge(0,1)
     m_0.remove_edge(n/2,n/2+1)
     n_0 = l_0.copy()
     n_0.remove_edge(2,3)
     n_0.remove_edge(n/2+2,n/2+3)
     m_1 = l_1.copy()
     n_1 = l_1.copy()
     return m_0, m_1, n_0, n_1;
     
###############################################################################

# This function is simply to depict the connectivity of the random circuit,
# it is not intended to actually generate a random state.
def display_random_tDesign(n, depth):
    l_0, l_1, pos_0, pos_1 = graph_layers(n)
    qreg = QuantumRegister(n, 'qreg')
    circuit = QuantumCircuit(qreg)
    for i in range(depth):
        for edge in l_0.edges():
            circuit.cz(qreg[edge[0]],qreg[edge[1]])
        for edge in l_1.edges():
            circuit.cz(qreg[edge[0]],qreg[edge[1]])
    return circuit;
    
###############################################################################

# This function will allow us to analyze how well the cutable circuits approximate
# Real random circuits
# patition the system by giving a list of lists
def product_test(circuit, partition, numshots):
    circ_inst = circuit.to_instruction()
    qnew0 = QuantumRegister(len(circuit.qubits), 'qnew0')
    qnew1 = QuantumRegister(len(circuit.qubits), 'qnew1')
    l0 = []
    l1 = []
    for i in range(len(qnew0)):
        l0.append(qnew0[i])
        l1.append(qnew1[i])
    qmeas = QuantumRegister(len(partition), 'qmeas')
    creg = ClassicalRegister(len(partition), 'creg')
    newCirc = QuantumCircuit(qmeas, qnew0, qnew1, creg)
    newCirc.append(circ_inst, l0)
    newCirc.append(circ_inst, l1)
    newCirc.h(qmeas)
    for i in range(len(partition)):
        for j in partition[i]:
            newCirc.cswap(qmeas[i],qnew0[j],qnew1[j])
    newCirc.h(qmeas)
    newCirc.measure(qmeas,creg)
    ex = qiskit.execute(newCirc, backend, shots =numshots)
    res = ex.result()
    counts = res.get_counts(newCirc)
    passkey = ''
    for i in range(len(partition)):
        passkey = passkey + '0'
    pTest = counts[passkey]/numshots
    return pTest;

###############################################################################

# Again, this is for display purposes only, not intended for computation.
def display_cutable_tDesign(n, depth):
    m_0, m_1, n_0, n_1 = cutable_graph_layers(n)
    qreg = QuantumRegister(n, 'qreg')
    circuit = QuantumCircuit(qreg)
    for i in range(depth):
        for edge in m_0.edges():
            circuit.cz(qreg[edge[0]],qreg[edge[1]])
        for edge in m_1.edges():
            circuit.cz(qreg[edge[0]],qreg[edge[1]])
    for i in range(depth):
        for edge in n_0.edges():
            circuit.cz(qreg[edge[0]],qreg[edge[1]])
        for edge in n_1.edges():
            circuit.cz(qreg[edge[0]],qreg[edge[1]])
    return circuit;

###############################################################################

def random_tDesign(n, depth):
    l_0, l_1, pos_0, pos_1 = graph_layers(n)
    qreg = QuantumRegister(n, 'qreg')
    circuit = QuantumCircuit(qreg)
    for i in range(depth):
        for edge in l_0.edges():
            U = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G = U.to_instruction()
            circuit.append(G, qargs = [qreg[edge[0]],qreg[edge[1]]])
        for edge in l_1.edges():
            U = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G = U.to_instruction()
            circuit.append(G, qargs = [qreg[edge[0]],qreg[edge[1]]])
    return circuit;

def cutable_tDesign(n, depth):
    m_0, m_1, n_0, n_1 = cutable_graph_layers(n)
    qreg = QuantumRegister(n, 'qreg')
    circuit = QuantumCircuit(qreg)
    for i in range(depth):
        for edge in m_0.edges():
            U = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G = U.to_instruction()
            circuit.append(G, qargs = [qreg[edge[0]],qreg[edge[1]]])
        for edge in m_1.edges():
            U = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G = U.to_instruction()
            circuit.append(G, qargs = [qreg[edge[0]],qreg[edge[1]]])
    for i in range(depth):
        for edge in n_0.edges():
            U = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G = U.to_instruction()
            circuit.append(G, qargs = [qreg[edge[0]],qreg[edge[1]]])
        for edge in n_1.edges():
            U = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G = U.to_instruction()
            circuit.append(G, qargs = [qreg[edge[0]],qreg[edge[1]]])
    return circuit;

# This will just be to demonstrate how close 1-local random unitaries are to fully ramdom ones.
def local_random(n):
    qreg = QuantumRegister(n,'qreg')
    circuit = QuantumCircuit(qreg)
    for i in range(n):
        theta = random.uniform(0,pi)
        phi = random.uniform(0,2*pi)
        lamb = random.uniform(0,2*pi)
        circuit.u3(theta,phi,lamb,qreg[i])
    return circuit;

def bipartite_random(n, depth):
    half_n = int(n/2)
    l_0, l_1, pos_0, pos_1 = graph_layers(half_n)
    qreg = QuantumRegister(n, 'qreg')
    circuit = QuantumCircuit(qreg)
    for i in range(depth):
        for edge in l_0.edges():
            U_0 = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G_0 = U_0.to_instruction()
            circuit.append(G_0, qargs = [qreg[edge[0]],qreg[edge[1]]])
            U_1 = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G_1 = U_1.to_instruction()
            circuit.append(G_1, qargs = [qreg[edge[0]+half_n],qreg[edge[1]+half_n]])
        for edge in l_1.edges():
            U_0 = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G_0 = U_0.to_instruction()
            circuit.append(G_0, qargs = [qreg[edge[0]],qreg[edge[1]]])
            U_1 = qiskit.quantum_info.random.utils.random_unitary(2**2)
            G_1 = U_1.to_instruction()
            circuit.append(G_1, qargs = [qreg[edge[0]+half_n],qreg[edge[1]+half_n]])
    return circuit;
        
###############################################################################
    
def chiSquared(dist1,dist2):
    chi=0
    for key in dist1.keys():
        if key in dist2.keys():
            chi=chi+((dist1[key]-dist2[key])**2/dist1[key])
        else:
            chi = chi + dist1[key]
    return chi;

def outcomes(numqubits):
    if numqubits==0:
        return [''];
    else:
        newStrings=[]
        s=str(0)
        for oldstring in outcomes(numqubits-1):
            oldstring=s+oldstring
            newStrings.append(oldstring)
        s=str(1)
        for oldstring in outcomes(numqubits-1):
            oldstring=s+oldstring
            newStrings.append(oldstring)
        return newStrings;

# Returns a library of uniform probabilities for n-qubit measurments
def uniform(n):
    probs = {}
    outs = outcomes(n)
    for out in outs:
        probs.update({out: 1/(2**n)})
    return probs;

################################################################################

# numshot_0 is the number of shots for each random circuit and numshot_1 is the number of
# times new randomness is generated
def verify_tDesign_randomness(n, depth, numshots_0, numshots_1):
    outs = outcomes(n)
    tDesign_counts = {}
    for out in outs:
        tDesign_counts.update({out : 0})
    tDesign_probs = {}
    for i in range(numshots_1):
        circuit = random_tDesign(n, depth)
        qreg = circuit.qubits
        creg = ClassicalRegister(n, 'creg')
        circuit.add_register(creg)
        circuit.measure(qreg,creg)
       # print(circuit)
        ex = qiskit.execute(circuit, backend, shots = numshots_0)
        res = ex.result()
        counts = res.get_counts(circuit)
      #  print(counts)
        for key in counts.keys():
            tDesign_counts.update({key : tDesign_counts[key] + counts[key]})
   # print(tDesign_counts)
    for key in tDesign_counts.keys():
        tDesign_probs.update({key: tDesign_counts[key]/(numshots_0*numshots_1)})
    chi = chiSquared(uniform(n), tDesign_probs)
   # print(tDesign_probs)
    
    return chi;

def verify_cutable_randomness(n, depth, numshots_0, numshots_1):
    outs = outcomes(n)
    tDesign_counts = {}
    for out in outs:
        tDesign_counts.update({out : 0})
    tDesign_probs = {}
    for i in range(numshots_1):
        circuit = cutable_tDesign(n, depth)
        qreg = circuit.qubits
        creg = ClassicalRegister(n, 'creg')
        circuit.add_register(creg)
        circuit.measure(qreg,creg)
       # print(circuit)
        ex = qiskit.execute(circuit, backend, shots = numshots_0)
        res = ex.result()
        counts = res.get_counts(circuit)
      #  print(counts)
        for key in counts.keys():
            tDesign_counts.update({key : tDesign_counts[key] + counts[key]})
   # print(tDesign_counts)
    for key in tDesign_counts.keys():
        tDesign_probs.update({key: tDesign_counts[key]/(numshots_0*numshots_1)})
    chi = chiSquared(uniform(n), tDesign_probs)
   # print(tDesign_probs)
    return chi;

def verify_local_randomness(n, numshots_0, numshots_1):
    outs = outcomes(n)
    local_counts = {}
    for out in outs:
        local_counts.update({out : 0})
    local_probs = {}
    for i in range(numshots_1):
        circuit = local_random(n)
        qreg = circuit.qubits
        creg = ClassicalRegister(n, 'creg')
        circuit.add_register(creg)
        circuit.measure(qreg,creg)
        ex = qiskit.execute(circuit, backend, shots = numshots_0)
        res = ex.result()
        counts = res.get_counts(circuit)
        for key in counts.keys():
            local_counts.update({key : local_counts[key] + counts[key]})
    for key in local_counts.keys():
        local_probs.update({key: local_counts[key]/(numshots_0*numshots_1)})
    chi = chiSquared(uniform(n), local_probs)
    return chi;

def verify_bipartite_randomness(n, depth, numshots_0, numshots_1):
    outs = outcomes(n)
    bipartite_counts = {}
    for out in outs:
        bipartite_counts.update({out : 0})
    bipartite_probs = {}
    for i in range(numshots_1):
        circuit = bipartite_random(n, depth)
        qreg = circuit.qubits
        creg = ClassicalRegister(n, 'creg')
        circuit.add_register(creg)
        circuit.measure(qreg,creg)
        ex = qiskit.execute(circuit, backend, shots = numshots_0)
        res = ex.result()
        counts = res.get_counts(circuit)
        for key in counts.keys():
            bipartite_counts.update({key : bipartite_counts[key] + counts[key]})
    for key in bipartite_counts.keys():
        bipartite_probs.update({key: bipartite_counts[key]/(numshots_0*numshots_1)})
    chi = chiSquared(uniform(n), bipartite_probs)
    return chi;

#####################################################################################

#print(product_test(random_tDesign(8,20), [[3,4,5,6],[7,0,1,2]], 1000))
#print(product_test(cutable_tDesign(8,10), [[3,4,5,6],[7,0,1,2]], 1000))
#print(product_test(bipartite_random(8,2), [[0,1,2,3],[4,5,6,7]], 1000))


#print('tDesign_chi', verify_tDesign_randomness(8, 6, 1000, 1000))
#print('cutable_tDesign_chi',verify_tDesign_randomness(8, 3, 1000, 1000))
#print('local_chi', verify_local_randomness(8,1000, 1000))
#print('bipartite_chi:', verify_bipartite_randomness(8,6,1000,1000))
    
#for i in range(1,8):
#    print('Random Depth:', i,  'chi_squared:', verify_tDesign_randomness(8, i, 1000, 500))

#print(uniform(8))



