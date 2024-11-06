# https://github.com/physicsisawesome7/Quantum_suduko_quanta/blob/e1a25c041e770ec091c638604e8054884d977c28/Implementation_code.py
from qiskit import *
import numpy as np
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from operator import itemgetter


a_string = QuantumRegister(1, name='a string')
b_string = QuantumRegister(1, name='b string')
c_string = QuantumRegister(1, name='c string')
d_string = QuantumRegister(1, name='d string')

all_0_ancilla_ab = QuantumRegister(2, name='all 0  ancilla ab')
all_0_ancilla_cd = QuantumRegister(2, name='all 0  ancilla cd')
all_0_ancilla_ac = QuantumRegister(2, name='all 0  ancilla ac')
all_0_ancilla_bd = QuantumRegister(2, name='all 0  ancilla bd')

out = QuantumRegister(1,name='output')

classical = ClassicalRegister(4, name='measure')

qc = QuantumCircuit(a_string, b_string, c_string, d_string, all_0_ancilla_ab, 
    all_0_ancilla_cd, all_0_ancilla_ac, all_0_ancilla_bd, out, classical)

#out in -
qc.x(out)
qc.h(out)

#superposition of input
qc.h(a_string[0])
qc.h(b_string[0])
qc.h(c_string[0])
qc.h(d_string[0])

qc.barrier()


def oracle(qc):

    # comparator ab
    qc.x(b_string[0])
    qc.mct([a_string[0], b_string[0]], all_0_ancilla_ab[0])
    qc.x(a_string[0])
    qc.x(b_string[0])
    qc.mct([a_string[0], b_string[0]], all_0_ancilla_ab[1])
    qc.x(a_string[0])
    qc.cnot(all_0_ancilla_ab[0], all_0_ancilla_ab[1])
    qc.barrier()

    # comparator ac
    qc.x(c_string[0])
    qc.mct([a_string[0], c_string[0]], all_0_ancilla_ac[0])
    qc.x(a_string[0])
    qc.x(c_string[0])
    qc.mct([a_string[0], c_string[0]], all_0_ancilla_ac[1])
    qc.x(a_string[0])
    qc.cnot(all_0_ancilla_ac[0], all_0_ancilla_ac[1])
    qc.barrier()

    # comparator cd
    qc.x(d_string[0])
    qc.mct([d_string[0], c_string[0]], all_0_ancilla_cd[0])
    qc.x(c_string[0])
    qc.x(d_string[0])
    qc.mct([d_string[0], c_string[0]], all_0_ancilla_cd[1])
    qc.x(c_string[0])
    qc.cnot(all_0_ancilla_cd[0], all_0_ancilla_cd[1])
    qc.barrier()

    # comparator bd
    qc.x(b_string[0])
    qc.mct([d_string[0], b_string[0]], all_0_ancilla_bd[0])
    qc.x(d_string[0])
    qc.x(b_string[0])
    qc.mct([d_string[0], b_string[0]], all_0_ancilla_bd[1])
    qc.x(d_string[0])
    qc.cnot(all_0_ancilla_bd[0], all_0_ancilla_bd[1])
    qc.barrier()


def diffuser(qc):

    qc.h(a_string[0])
    qc.h(b_string[0])
    qc.h(c_string[0])
    qc.h(d_string[0])

    qc.x(a_string[0])
    qc.x(b_string[0])
    qc.x(c_string[0])
    qc.x(d_string[0])

    qc.h(a_string[0])
    qc.mct([ b_string[0], c_string[0], d_string[0]], a_string[0])
    qc.h(a_string[0])

    qc.x(a_string[0])
    qc.x(b_string[0])
    qc.x(c_string[0])
    qc.x(d_string[0])

    qc.h(a_string[0])
    qc.h(b_string[0])
    qc.h(c_string[0])
    qc.h(d_string[0])

    qc.barrier()


#n is the number of iteration of Grover operator
n = 2

for i in range (n):

    oracle(qc)
    qc.mct([all_0_ancilla_ab[1], all_0_ancilla_ac[1],all_0_ancilla_cd[1], 
        all_0_ancilla_bd[1]], out)
    qc.barrier()

    oracle(qc)
    diffuser(qc)

qc.measure(a_string[0],classical[0])
qc.measure(b_string[0],classical[1])
qc.measure(c_string[0],classical[2])
qc.measure(d_string[0],classical[3])

print(qc)

def print_sudoku(dict):
    list=[]
    for item in dict:
        for a in item:
            list.append(a)
        data=np.array(list)
        shape=(2,2)
        sudoku=data.reshape(shape)
        print(sudoku)
        print('\n')
        list.clear()

#start simulation
backend = QasmSimulator()
result = execute(qc, backend=backend, shots=1024).result()
answer = result.get_counts()
plot_histogram(answer).show()

#return N max values
res = dict(sorted(answer.items(), key=itemgetter(1), reverse=True)[:2])

#print the sudoku schemas
print('the solutions are:')
print_sudoku(res)
