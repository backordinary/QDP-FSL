# https://github.com/huub-d96/QA_Stage/blob/75ced905b5c87fcb43f67af7894704cc4b21ff9b/cirq_decomp.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:09:02 2021

@author: huub
"""

import cirq
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import xacc
import QAOA
import generate_graph as gg
from math import pi, sqrt, acos
import networkx as nx

qpu_id = 'ibm:ibmq_qasm_simulator'
qpu = xacc.getAccelerator(qpu_id, {'shots':2048})

g = gg.regular_graph(5)
graph = nx.Graph()
graph.add_nodes_from(range(5))
graph.add_edges_from(g[1])

# graph = gg.tsp_problem_set(3, gg.regular_graph) 
params = [1.0, 1.0]*1

xacc_circuit = QAOA.genDSPCircuit(qpu, qpu_id, graph, params)

    
        
# additional gates
def cry(theta):
    qc = QuantumCircuit(2)
  #  qc.cnot(0, 1)
  #  qc.ry(-theta/2, 1)
  #  qc.cnot(0, 1)
  #  qc.ry(theta / 2, 1)

    qc.ry((pi/2)-theta/2, 0)
    qc.cnot(1, 0)
    qc.ry(-((pi/2)-theta/2), 0)

    return qc.to_gate()


def ccry(theta):
    qc = QuantumCircuit(3)
    qc.toffoli(0, 1, 2)
    qc.ry(-theta/2, 2)
    qc.toffoli(0, 1, 2)
    qc.ry(theta / 2, 2)
    return qc.to_gate()


def scs(n, k):
    qc = QuantumCircuit(n)
    qc.cnot(n - 2, n-1)
    theta = 2 * (acos(sqrt(1 / n)))
    qc.append(cry(theta), [n-2, n - 1])
    qc.cnot(n - 2, n-1)

    for m in range(k - 1):
        l = 2+m
        control = n-2-m
        qc.cnot(control-1, n-1)
        theta = 2 * (acos(sqrt((n-control) / n)))
        qc.append(ccry(theta), [n-1, control, control-1])

        qc.cnot(control-1, n-1)
    return qc.to_gate()


def dicke_init(n, k):
    #deterministic  Dicke state preparation (Bärtschi & Eidenbenz, 2019)
    #unoptimized version
    qc = QuantumCircuit(n)
    qc.x(range(n-k, n))
    for i in range(n, k, -1):
        qc.append(scs(i, k), range(0, i))

    for i in range(k, 1, -1):
        qc.append(scs(i, i-1), range(0, i))

    return qc.to_gate()

def OR_2q():
    qc = QuantumCircuit(3)
    qc.toffoli(0, 1, 2)
    qc.cnot(0, 1)
    qc.cnot(0, 2)
    return qc.to_gate()


def OR_nrz(n, gamma):
    ORGate = OR_2q()
    qc = QuantumCircuit(2*n)
    qc.append(ORGate, [0, 1, n])
    for i in range(2, n):
        qc.append(ORGate, [i, n+i-2, n+i-1])
    qc.crz(gamma, 2*n-2, 2*n-1)
    for i in range(n, 2, -1):
        qc.append(ORGate, [n-i-1, 2*n-2-i, 2*n-1-i])
    qc.append(ORGate, [0, 1, n])
    return qc.to_gate()

####TSP#####
# p = len(params)//2
# beta = params[0:p]
# gamma = params[p:2*p]
# v, A, D = graph

# vertice_list = list(range(0, v, 1))
    
# n = v**2
# qc = QuantumCircuit(n, n)
# # INIT  :    This initiation can and should be reused
# for q in range(v):
#     q_range = range(q*v, (q+1)*v)
#     DickeGate = dicke_init(v, 2)
#     qc.append(DickeGate, q_range)
# # QAOA cycles
# #for 1 to p:

# #cost unitary
# for it in range(p):
#     for i in range(n):
#         qc.rz(gamma[it]*D[i]/(2*pi), i)

#     for i in range(v):
#         for j in range(i):
#             if i != j:
#                 qc.rzz(20*gamma[it]/pi, (j+i*v), i+j*v)   #RZZ on reflection over the diagonal

#     #mixer unitary
#     for i in range(0, v):
#         qc.rxx(-beta[it], i*v, (i*v+1))
#         qc.rxx(-beta[it], (i*v+1), (i*v+2))

#         qc.ryy(-beta[it], i * v, (i * v + 1))
#         qc.ryy(-beta[it], (i * v + 1), (i * v + 2))


# qc.measure(range(n), range(n))

####DSP#####
p = len(params)//2
beta = params[0:p]
gamma = params[p:2*p]
v = graph.number_of_nodes()
edge_list = [list(edge) for edge in graph.edges()]
vertice_list = list(range(0, v, 1))

connections = []
for i in range(v):
    connections.append([i])
for t in edge_list:
    connections[t[0]].append(t[1])
    connections[t[1]].append(t[0])
ancillas = 0
for con in connections:
    if len(con) > ancillas:
        ancillas = len(con)
n = v+ancillas         # add ancillas
qc = QuantumCircuit(n, v)

for qubit in range(v):
    qc.h(qubit)
    #inverted  crz gate
    qc.x(qubit)
    qc.crz(-gamma[0], qubit, n-1)
    qc.x(qubit)
for iteration in range(p):
    f = 0
    f_anc = v
    for con in connections:  # TODO: fix for unordered edges e.g. (2,0)
        c_len = len(con)
        OR_range = con.copy()
        for k in range(c_len-1):
            OR_range.append(v+k)
        cOR_rz = OR_nrz(c_len, gamma[p-1])
        OR_range.append(n-1)
        qc.append(cOR_rz, OR_range)

    for qb in vertice_list:
        qc.rx(-2*beta[p-1], qb)
qc.measure(range(v), range(v))

decomp = qc.decompose()#.decompose().decompose().decompose().decompose()

decomp.draw(output='mpl')

#from qiskit import QuantumCircuit
#qiskit_qc = QuantumCircuit.from_qasm_str(qpu.getNativeCode(xacc_circuit, {'format': 'qasm'}))
#print(qiskit_qc)
#qiskit_qc.draw('mpl')        