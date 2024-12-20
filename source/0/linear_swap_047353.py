# https://github.com/ZeX2/Qvandidat/blob/7be5532f4e8b83ef656956a2074dd7beb503a698/routing_methods/linear_swap.py
import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from .decompose_circuit import decompose_qaoa_circuit

def swapPositions(list, pos1, pos2):  
    list[pos1], list[pos2] = list[pos2], list[pos1] 
    return list

def UL_swap(array):
    N = len(array)
    for i in range(0,N-1,2):
        array = swapPositions(array, i, i+1)
    return array 

def UR_swap(array):
    N = len(array)
    for i in range(1,N-2,2):
        array=swapPositions(array, i+1, i+2)
    return array 

def do_all_ops(circuit, qubit_line, qubit_path, operations):
    didzz = False
    M = len(qubit_line)
    # The order is to reduce circuit depth
    # so do not simplify the for loops because
    # 
    n = 2
    if M < 4:
        n = 1
    for i in range(0,M-1,n):
        logical_q1 = qubit_path[i]
        logical_q2 = qubit_path[i+1]
        q1 = qubit_line[i]
        q2 = qubit_line[i+1]
        didzz |= do_op_new(q1, q2, logical_q1, logical_q2, circuit, operations)

    for i in range(1,M-1,n):
        logical_q1 = qubit_path[i]
        logical_q2 = qubit_path[i+1]
        q1 = qubit_line[i]
        q2 = qubit_line[i+1]
        didzz |= do_op_new(q1, q2, logical_q1, logical_q2, circuit, operations)
    return didzz

def do_op_new(q1, q2, logical_q1, logical_q2, circuit, operations):
    didzz = False
    if operations[logical_q1,logical_q2] != 0:
        do_zz_op(circuit, q1, q2, operations[logical_q1,logical_q2])
        didzz = True

    if operations[logical_q2,logical_q1] != 0:
        do_zz_op(circuit, q2, q1, operations[logical_q2,logical_q1])
        didzz = True

    operations[logical_q1,logical_q2] = 0
    operations[logical_q2,logical_q1] = 0

    return didzz

def do_zz_op(circuit, qubit1, qubit2, angle):
    #circuit.barrier()
    circuit.rzz(angle,qubit1,qubit2)
    #circuit.barrier()

def do_zz_op2(circuit, qubit1, qubit2, angle):
    #circuit.barrier()
    circuit.cx(qubit1,qubit2)
    circuit.rz(angle,qubit2)
    circuit.cx(qubit1,qubit2)
    #circuit.barrier()

    #print('zz \\w', qubit1, 'and', qubit2)

def qc_UL_swap(circuit, qubit_path, qubit_line):
    N = len(qubit_path)

    for i in range(0,N-1,2):
        qubit_path = swapPositions(qubit_path, i, i+1)
        circuit.swap(qubit_line[i], qubit_line[i+1])
    return circuit 

def qc_UR_swap(circuit, qubit_path, qubit_line):
    N = len(qubit_path)
    for i in range(1,N-1,2):
        qubit_path = swapPositions(qubit_path, i, i+1)
        circuit.swap(qubit_line[i], qubit_line[i+1])
    return circuit 

def qc_UL_UR(input_circuit, qubit_line, qubit_path, operations):

    circuit = input_circuit.copy()
    num_qubits = len(qubit_path)

    for i in range(int(num_qubits/2)):
        
        if np.any(operations):
            qc_UL_swap(circuit, qubit_path, qubit_line)
        do_all_ops(circuit, qubit_line, qubit_path, operations)
        
        if np.any(operations):
            qc_UR_swap(circuit, qubit_path, qubit_line)
        do_all_ops(circuit, qubit_line, qubit_path, operations)
    return circuit


def get_operations(J, gamma):
    N = len(J)
    operations = np.zeros(J.shape)
    for i in range(N):
        for j in range(i):
            operations[i,j] = 2*gamma*J[i,j]
    return operations.T


# J and qubit_path has to have 2**k qubits
def linear_swap_method_outdated(J, gamma, beta, qubit_line = None):
    N = len(J)
    operations = get_operations(J, gamma)
    #print(operations)
    circuit = QuantumCircuit(N)
    circuit.h(range(N))
    circuit.barrier()
    
    
    if qubit_line is None:
        qubit_line = np.array(range(N))
    qubit_path = qubit_line.copy()

    do_all_ops(circuit, qubit_line, qubit_path, operations)
    new_circ = qc_UL_UR(circuit, qubit_line, qubit_path, operations)

    new_circ.rx(2*beta, range(N))
    #print(operations)
    for i in range(N):
        qq = qubit_path[i]
        new_circ.measure(i, qq)
    return new_circ

def linear_swap_method(qaoa_circuit, p, qubit_line=None):
    N = qaoa_circuit.num_qubits
    operations, rz_ops, rx_ops = decompose_qaoa_circuit(qaoa_circuit, N, p)
    circuit = QuantumCircuit(N, N)
    circuit.h(range(N))
    
    if qubit_line is None:
        qubit_line = np.array(range(N))
    qubit_path = qubit_line.copy()
    
    for i in range(p):
        do_all_ops(circuit, qubit_line, qubit_path, operations[i,:,:])
        circuit = qc_UL_UR(circuit, qubit_line, qubit_path, operations[i,:,:])
        
        for q, theta in enumerate(rz_ops[i,:]):
            qq = qubit_path[q]
            circuit.rz(theta, qq)

        for q, theta in enumerate(rx_ops[i,:]):
            qq = qubit_path[q]
            circuit.rx(theta, qq)

    for q, theta in enumerate(rx_ops[0,:]):
        qq = qubit_path[q]
        circuit.measure(q, qq)
    
    return circuit

def simplify(circuit):
    return transpile(circuit, basis_gates=['cz', 'rz', 'u3', 'iswap'], optimization_level=3)

