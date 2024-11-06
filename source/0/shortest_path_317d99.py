# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/shortest%20path/shortest_path.py
import numpy as np
import math

# importing Qiskit
import qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer

# import basic plot tools
from qiskit.visualization import plot_histogram
d=1
n=2
dreg = QuantumRegister(d, 'd')
nreg = QuantumRegister(n, 'n')
creg = ClassicalRegister(n, 'c')
circ = QuantumCircuit(dreg, nreg, creg)
found_path = False

nodes = ["00","01","10","11"]
starting_node = nodes[0]
end_node = nodes[3]
current_node = starting_node
vertics =   [[0,1,1,0],
            [1,0,1,1],
            [1,1,0,1],
            [0,1,1,0]]
numb_vertices = len(vertics[0])
print("Starting node ", starting_node, " has edges to: ", vertics[0])

vistied_nodes = []
def init_circuit(a):
    cnt = 1
    for x in zip(a):
        if x == "0":
            pass
        else:
            circ.x(cnt)
        cnt +=1

def compare(a, b):
    i = 1
    for x, y in zip(a, b):

        if x == str(y):
            #return True
            #print("Pos ", i, "stimmt überein")
            pass
        else:
            #return False
            #print("Pos ", i, "stimmt nicht überein")
            circ.cx(0,i)
        i =i+1
    circ.x(0)

possible_vertics = []    
j = 1
while end_node != current_node:
    #while found_path == False:
    init_circuit(current_node)
    circ.h(0)
    for i in range(numb_vertices):
        if vertics[0][i] == 1:
            #print("We can go to vertics: ", nodes[i])
            compare(current_node, nodes[i])

    # Execute and see results
    circ.measure(nreg[:n],creg)
    backend = BasicAer.get_backend('qasm_simulator')
    shots = 1
    results = execute(circ, backend=backend, shots=shots).result()
    answer = results.get_counts()
    for key, value in answer.items():
        vistied_nodes.append(key)
        print("Currently in iteration no. ", j, "We are at node ", key)
        current_node = key
        if key == "11":
            found_path=True
    j = j+1
    input("Happy?")
print("we found the path", vistied_nodes)