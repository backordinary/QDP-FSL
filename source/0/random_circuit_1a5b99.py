# https://github.com/echo724/qunix/blob/21bf50c2619decdf7cde41b1e062bc5f3147ffa0/qsay/random_circuit.py
#!/usr/bin/env python3

def warn (* args, **kwargs):
    pass
import warnings
warnings.warn = warn

from qiskit import *
import numpy as np
import argparse
import sys
import os

def q_rand_number(q_num, c_num):
    
    circuit = QuantumCircuit(q_num,c_num)
    for i in range(q_num):
        circuit.h(i)
        circuit.measure([i], [i])

    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend = simulator, shots = 4, memory=True).result()

    # h, x, y, z, m, barrier, rx, rz, ry, cx, crx
    gate_num = int(result.get_memory()[0], 2) % 11 

    # location of gate
    gate_loc = int(result.get_memory()[1], 2) % q_num

    # angle
    if int(result.get_memory()[2], 2) is not 0:
        angle = np.pi / int(result.get_memory()[2], 2)
    else:
        angle = 0

    # location of connection
    target_loc = int(result.get_memory()[0], 2) % q_num

    # rx, rz, ry, cx, crx:
    if gate_num > 8:
        while gate_loc == target_loc:
            result = execute(circuit, backend = simulator, shots = 1, memory=True).result()
            target_loc = int(result.get_memory()[0], 2) % q_num

    return gate_num, gate_loc, angle, target_loc


def build_random_circuit(q_num, size):
    c_num = q_num
    circuit = QuantumCircuit(q_num,c_num)
    
    for i in range(size):
        
        # generate random gate
        gate_num, gate_loc, angle, target_loc = q_rand_number(q_num, c_num)
        
        # h, x, y, z, m, barrier, rx, rz, ry, cx, crx
        if gate_num==0:
            circuit.h(gate_loc)
        elif gate_num==1:
            circuit.x(gate_loc)
        elif gate_num==2:
            circuit.y(gate_loc)
        elif gate_num==3:
            circuit.z(gate_loc)
        elif gate_num==4:
            circuit.measure([gate_loc], [gate_loc])
        elif gate_num==5:
            circuit.barrier()
        elif gate_num==6:
            circuit.rx(angle, gate_loc)
        elif gate_num==7:
            circuit.rz(angle, gate_loc)
        elif gate_num==8:
            circuit.ry(angle, gate_loc)
        elif gate_num==9:
            circuit.cx(gate_loc, target_loc)
        elif gate_num==10:
            circuit.crx(angle, gate_loc, target_loc)
    
    # draw circuit
    # print(circuit.draw(output='text'))
    return circuit.draw(output='text')
 
def comment_feeling(quote):
    splitted = quote.split("\n")
    nextline = ""
    longest = 0
    for line in splitted:
        line += nextline
        if len(line) > longest and len(line) < 75:
            longest = len(line)
        if len(line) > 74:
            longest = 75
            nextline = line[75:]
    longest += 1
    print("","-"*longest," ")
    for i,line in enumerate(splitted):
        line += nextline
        nextline = ""
        if len(line) > 74:
            nextline = line[75:]
            line = line[:74]
        print("|","",end="")
        print(line,end="")
        print(" "*(longest - len(line) - 1),"|")
    while len(nextline) > 0:
        print("|","",end="")
        remain=len(nextline)
        if remain > 75:
            print(nextline[:75],end="")
            print("","|")
            nextline = nextline[75:]
        else:
            print(nextline,end="")
            print(" "*(longest - len(nextline) - 1),"|")
            break
    print("","-"*longest," ")
    print("")


parser = argparse.ArgumentParser(description = '''
--------------------------------------------------
Generate random circuit
Enter the number of qubits and the number of gates
(number of gates includes barrier)
default: -q 5 -s 5
--------------------------------------------------
''')

# get number of qubits
parser.add_argument('-q', type=int, default=5)

# get number of gates
parser.add_argument('-s', type=int, default=5)

# get string
parser.add_argument('-quote', type=str, default="Random Curcuit")
args = parser.parse_args()

comment_feeling(args.quote)
print(build_random_circuit(args.q,args.s))



