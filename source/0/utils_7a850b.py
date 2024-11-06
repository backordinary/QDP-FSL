# https://github.com/jackkyyh/CNOT-Qiskit/blob/79d5cce3359a9a9c3f887a7cf1de42a8755dc4d9/utils.py

import numpy as np
from qiskit import QuantumCircuit
# from qiskit import *
from qiskit import Aer, execute

backend = Aer.get_backend('unitary_simulator')

def add_row(A, op):    
    op = np.array(op)
    for i, j in op:
        A[j] = (A[i] + A[j]) % 2
    return A



def parse_circ_qasm(file):
    with open(file, 'r') as f:
       lines = f.readlines()

    num_qubits = int(lines[0])
    circ = [num_qubits]

    for line in lines[1:]:
        line = line.strip().split(' ')
        if(line[0] == 'CNOT'):
            gate, qu1, qu2 = line
            qu1, qu2 = int(qu1), int(qu2)
            circ.append(('CNOT', qu1, qu2))
        else:
            # continue
            gate, qu = line
            qu = int(qu)
            circ.append((gate, qu))

    return circ


def parse_circ_sarah(file):
    with open(file, 'r') as f:
       lines = f.readlines()

    num_qubits = 9
    circ = [num_qubits]
    
    for line in lines[6:]:
        line = line.strip().split('[')
        if(line[0] == 'CNOT'):
            qu1, qu2 = int(line[1][0])-1, int(line[1][3])-1
            circ.append(('CNOT', qu1, qu2))
        else:
            # continue
            gate, qu = line
            # print(line)
            qu = int(qu[0])-1
            circ.append((gate, qu))

    # circ.draw()
    return circ


def parse_circ_sarah2(file):
    with open(file, 'r') as f:
       lines = f.readlines()

    num_qubits = 9
    circ = [num_qubits]
    
    for line in lines:
        q1, q2 = int(line[1])-1, int(line[4])-1
        circ.append(('CNOT', q1, q2))

    # circ.draw()
    return circ


def build_circ(parsed_circ, circ_type=QuantumCircuit):
    num_qubits = parsed_circ[0]
    circ = circ_type(num_qubits, 0)

    gate_map = {'X': circ.x, 'Y': circ.y, 'Z': circ.z,
        'H': circ.h, 'T': circ.t, 'T+': circ.tdg,
        'S': circ.s, 'S+': circ.sdg,
        'CNOT': circ.cnot}

    for gate_line in parsed_circ[1:]:
        gate, qu = gate_line[0], gate_line[1:]
        gate_map[gate](*qu)
    
    return circ


def build_parity_mtx(parsed_circ):
    A = np.identity(parsed_circ[0])

    for gate_line in parsed_circ[1:]:
        if(gate_line[0] == 'CNOT'):
            qu1, qu2 = gate_line[1], gate_line[2]
            # print(qu1, qu2)
            A[qu2] = (A[qu1] + A[qu2])%2
        
    return A



def get_unitary(circ):
    # circ.draw()
    job = execute(circ, backend)
    result = job.result()
    u = result.get_unitary(circ).data
    return u

