# https://github.com/M4D-A/PYQSAT/blob/080738c50969463a8aaba9c03d09697d3432c4e1/PYQSAT.py
#!/usr/bin/python3

import sys
import time
import numpy as np
import math as m
from qiskit import QuantumCircuit, transpile, Aer

def mcz(circuit, qubits):
    circuit.h(qubits[-1])
    circuit.mcx(qubits[:-1],qubits[-1])
    circuit.h(qubits[-1])

def open_CNF(filename):
    with open(filename, "r") as CNF_file:
        content = CNF_file.read()
    lines = content.split("\n")
    header = [line for line in lines if line and line[0] == 'p'][0].split(" ")
    v_num = int(header[2])
    c_num = int(header[3])
    clauses = [line for line in lines if line and line[0] != 'c' and line[0] != 'p']
    tempCNF = [clause.split(" ") for clause in clauses]
    CNF = [[int(literal) for literal in clause if literal != '0'] for clause in tempCNF]
    return (CNF, v_num, c_num)

def split_CNF(CNF):
    absCNF = [[abs(literal)-1 for literal in clause] for clause in CNF]
    posCNF = [[abs(literal)-1 for literal in clause if literal > 0] for clause in CNF]
    negCNF = [[abs(literal)-1 for literal in clause if literal < 0] for clause in CNF]
    return(absCNF, posCNF, negCNF)

def q_solve(CNF, V, C, S):
    absCNF, posCNF, negCNF = split_CNF(CNF)
    N = V+C
    V_range = list(range(V))
    C_range = list(range(V, N))
    r = m.pi / (4.0 * m.sqrt(S)) * m.pow(2.0,V/2.0)
    R = m.floor(r)

    simulator = Aer.get_backend('statevector_simulator')

    #//------------------// Definicja obwodu kwantowego //------------------//#
    circuit = QuantumCircuit(V+C, V)
    circuit.h(V_range)
    for _ in range(R):

        #//Wyrocznia Kwantowa//#
        for i, pos_cls, abs_cls in zip(C_range, posCNF, absCNF):
            if(pos_cls): circuit.x(pos_cls)
            circuit.mcx(abs_cls, i)
            circuit.x(i)
            if(pos_cls): circuit.x(pos_cls)

        mcz(circuit, C_range)

        for i, pos_cls, abs_cls in list(zip(C_range, posCNF, absCNF))[::-1]:
            if(pos_cls): circuit.x(pos_cls)
            circuit.mcx(abs_cls, i)
            circuit.x(i)
            if(pos_cls): circuit.x(pos_cls)
        #//------------------//#

        #//Dyfuzja Grovera//#
        circuit.h(V_range)
        circuit.x(V_range)
        mcz(circuit, V_range)
        circuit.x(V_range)
        circuit.h(V_range)
        #//---------------//#

    circuit.measure(V_range, V_range) #//POMIAR//#
    #//---------------------------------------------------------------------//#

    compiled_circuit = transpile(circuit, simulator) #Kompilowanie obwodu
    result = simulator.run(compiled_circuit).result() #Symulacja obwodu
    solution = list(result.get_counts().keys())[0][::-1] #Pobranie wyniku pomiaru

    vector = [] #Formatowanie wyniku do rozwiązania funkcji CNF w formacie DIMACS
    for i,b in enumerate(solution):
        literal = i+1 if b=="1" else -(i+1)
        vector.append(literal)
    return vector

def eval_clause(clause, solution):
    for literal in solution:
        if literal in clause:
            return True
    return False

def eval_solution(CNF, solution):
    for clause in CNF:
        if not eval_clause(clause, solution):
            return False
    return True

if __name__ == "__main__":
    start = time.time()
    CNF, V, C = open_CNF(sys.argv[1])
    S = 1 if len(sys.argv) < 3 else int(sys.argv[2])
    solution = q_solve(CNF, V, C, S)
    print("c <<PYQSAT>>")
    print("c Filename:",sys.argv[1],"clauses:",C,"variables:",V)
    print("c")
    if eval_solution(CNF, solution):
        print("s SATISFIABLE")
        print("v", end=" ")
        for literal in solution:
            print(literal, end=" ")
        print("0")
    else:
        print("s UNDETERMINED")
    print("c")
    print("c Time elapsed:", "{:5.2f}".format(time.time()-start), "s")
