# https://github.com/KangHaiYue/Laby-Research-Project-Quantum-computing/blob/644ed958f0905643857ec89de1c3a71d5dd27477/adder_byte.py
'''This file is created by Haiyue Kang 21 Jan 2022'''
'''The following codes builds a quantum adder that can add two integers with total no larger than 8 bit
(2^8)'''

from ibm_quantum_widgets import CircuitComposer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library.standard_gates import ZGate
import numpy as np

def input_translator(circ,x,y):
    '''takes 2 integer number in decimal, translate into binary, and implant them into the
    quantum circuit'''
    x_bin = bin(x)[2:]
    y_bin = bin(y)[2:]
    x_bin_lst = [int(i) for i in x_bin]
    y_bin_lst = [int(i) for i in y_bin]
    x_bin_lst.reverse()
    y_bin_lst.reverse()
    for i in range(len(x_bin_lst)):
        if x_bin_lst[i] == 1:
            circ.x(3*i)
    for i in range(len(y_bin_lst)):
        if y_bin_lst[i] == 1:
            circ.x(3*i+1)

def output_translator(x):
    '''translate a integer from binary to decimal'''
    sum_ = list(x.keys())[0]
    sum_ = int(sum_[:8],2)
    return sum_
    
def logical_and(circ, q0,q1,q2):
    '''defines logical AND gate (see https://quantum-journal.org/papers/q-2018-06-18-74/)'''
    circ.h(q2)
    circ.t(q2)
    circ.cx(q0,q2)
    circ.cx(q1,q2)
    circ.cx(q2,q0)
    circ.cx(q2,q1)
    circ.tdg(q0)
    circ.tdg(q1)
    circ.t(q2)
    circ.cx(q2,q0)
    circ.cx(q2,q1)
    circ.h(q2)
    circ.s(q2)

def uncomp_and(circ,q0,q1,q2,cbit):
    '''uncompute logical AND gate (see https://quantum-journal.org/papers/q-2018-06-18-74/)'''
    circ.h(q2)
    circ.measure(q2,cbit)
    double_ctrl_z = ZGate().control(2)
    circ.append(double_ctrl_z, [q0,q2,q1])

def adder(a,b):
    '''adds two integers quantum mechanically using Craig Gidney's adder algorithm, sum within 8 bits'''
    '''both inputs and outputs are just normal decimals, the function will translate them into
    and back from binary for you'''
    #creates the circuit
    qreg_q = QuantumRegister(23, 'q')
    creg_c = ClassicalRegister(15, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    #inintialize the inputs to the circuit
    input_translator(circuit,a,b)
    
    #calculates the sum of a and b using Craig's adder algorithm
    logical_and(circuit, qreg_q[0], qreg_q[1], qreg_q[2])
    for i in [0,3,6,9,12,15]:
        circuit.cx(qreg_q[i+2],qreg_q[i+3])
        circuit.cx(qreg_q[i+2],qreg_q[i+4])
        logical_and(circuit, qreg_q[i+3], qreg_q[i+4], qreg_q[i+5])
        circuit.cx(qreg_q[i+2],qreg_q[i+5])
    
    circuit.cx(qreg_q[20],qreg_q[22])
    
    for i in [15,12,9,6,3,0]:
        circuit.cx(qreg_q[i+2],qreg_q[i+5])
        uncomp_and(circuit, qreg_q[i+3], qreg_q[i+4], qreg_q[i+5],[15,12,9,6,3,0].index(i))
        circuit.cx(qreg_q[i+2],qreg_q[i+3])
    
    uncomp_and(circuit, qreg_q[0], qreg_q[1], qreg_q[2],3)
    for i in [0,3,6,9,12,15,18,21]:
        circuit.cx(qreg_q[i],qreg_q[i+1])
    #measure the result
    for i in [0,3,6,9,12,15,18,21]:
        circuit.measure(qreg_q[i+1],[0,3,6,9,12,15,18,21].index(i)+7)
    backend = QasmSimulator()
    qc_compiled = transpile(circuit, backend)
    job = backend.run(qc_compiled, shots = 1)
    result = job.result()
    counts = result.get_counts(qc_compiled)
    return(output_translator(counts))