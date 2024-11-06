# https://github.com/PierreEngelstein/MasterRecherche/blob/31eebf636d2025d4d6841119971a8a0d5b00f8ef/Programmation/GateBuilding/function_evaluate_quantum.py
'''
Implementing basic classical gates using quantum gates.
Both circuit built here are equivalent to
f(a, b, c, d) = ((a.b)+(a.!b.c)).((d^a)+(a^c^d))

The first implementation uses building blocks: AND, AND_NOT, OR, XOR and NOT which are implemented
using controlled-x gates.

The second implementation is an optimized one, using the method presented by Younes and Miller in 2003.
(available at https://cds.cern.ch/record/612920/files/0304099.pdf).
'''
from qiskit import *
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.library.standard_gates.z import ZGate

def AND():
    _circ = QuantumCircuit(3, name="AND")
    _circ.ccx(0, 1, 2)
    # _circ.append(ZGate().control(2), [0, 1, 2])
    return _circ.to_instruction()

def AND_NOT():
    ''' r = a && !b '''
    _circ = QuantumCircuit(3, name="AND_NOT")
    _circ.ccx(0, 1, 2)
    # _circ.append(ZGate().control(2), [0, 1, 2])
    _circ.cx(0, 2)
    # _circ.append(ZGate().control(1), [0, 2])
    return _circ.to_instruction()

def OR():
    _circ = QuantumCircuit(3, name="OR")
    # _circ.append(ZGate().control(1), [0, 2])
    # _circ.append(ZGate().control(1), [1, 2])
    # _circ.append(ZGate().control(2), [0, 1, 2])
    _circ.cx(0, 2)
    _circ.cx(1, 2)
    _circ.ccx(0, 1, 2)
    return _circ.to_instruction()

def XOR():
    _circ = QuantumCircuit(3, name="XOR")
    # _circ.append(ZGate().control(1), [0, 2])
    # _circ.append(ZGate().control(1), [1, 2])
    _circ.cx(0, 2)
    _circ.cx(1, 2)
    return _circ

def NOT():
    _circ = QuantumCircuit(1, name="NOT")
    # _circ.append(ZGate(), [0])
    _circ.x(0)
    return _circ.to_instruction()

def OptimizedCircuit():
    _circ = QuantumCircuit(5, name="CIRC")
    _circ.toffoli(0, 2, 4)
    _circ.append(XGate().control(4), [0, 1, 2, 3, 4])
    _circ.append(XGate().control(3), [0, 1, 2, 4])
    _circ.append(XGate().control(3), [0, 1, 3, 4])
    _circ.toffoli(0, 1, 4)

    # _circ.append(ZGate().control(2), [0, 2, 4])
    # _circ.append(ZGate().control(4), [0, 1, 2, 3, 4])
    # _circ.append(ZGate().control(3), [0, 1, 2, 4])
    # _circ.append(ZGate().control(3), [0, 1, 3, 4])
    # _circ.append(ZGate().control(2), [0, 1, 4])

    return _circ.to_instruction()


print("************")
print("BLOCK BY BLOCK CIRCUIT BUILD")
print("************")
qc = QuantumCircuit(13, 13)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.barrier()
qc.append(AND(), [0, 1, 4])
qc.append(AND_NOT(), [0, 1, 5])
qc.append(AND(), [2, 5, 6])
qc.append(OR(), [4, 6, 7])
qc.barrier()
qc.append(XOR(), [3, 0, 8])
qc.append(XOR(), [0, 2, 9])
qc.append(XOR(), [9, 3, 10])
qc.append(OR(), [8, 10, 11])
qc.barrier()
qc.append(AND(), [7, 11, 12])
qc.barrier()
qc.measure(0,  4)
qc.measure(1,  3)
qc.measure(2,  2)
qc.measure(3,  1)
qc.measure(12, 0)


qc.draw(output='mpl', filename='circuit.png')
circuit_decp = qc.decompose()
# circuit_decp.draw(output='mpl', filename='circuit.png')
# circuit_decp.draw(output='latex_source', filename='circuit.tex')

# target = qiskit.quantum_info.Statevector.from_instruction(qc)
# print(target)

result = qiskit.visualization.circuit_drawer(qc, output="text")
print(result)

result = qiskit.visualization.circuit_drawer(circuit_decp, output="text")
print(result)

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
result = qiskit.execute(qc, backend,shots=shots).result()
counts = result.get_counts(qc)
for res in sorted(counts.int_raw):
    print("{0:05b}".format(res))
print(counts)

print("************")
print("OPTIMIZED CIRCUIT BUILD")
print("************")

qc = QuantumCircuit(5, 5)
# qc.h(0)
# qc.h(1)
# qc.h(2)
# qc.h(3)
qc.barrier()
qc.append(OptimizedCircuit(), [0, 1, 2, 3, 4])
qc.barrier()
# qc.measure(0, 4)
# qc.measure(1, 3)
# qc.measure(2, 2)
# qc.measure(3, 1)
# qc.measure(4, 0)

import numpy as np
backend = Aer.get_backend('unitary_simulator')
job = execute(qc, backend)
result = job.result()
result_matrix = result.get_unitary(qc, decimals=3)
a = np.asarray(result_matrix)
np.savetxt("unitary.csv", a, delimiter=',')
print(result_matrix)

for i in result_matrix:
    for j in i:
        print(int(j.real), end=' ')
    print()

print("**********")

res_transpose = np.dot(np.transpose(result_matrix), result_matrix)
for i in res_transpose:
    for j in i:
        print(int(j.real), end=' ')
    print()

1/0

circuit_decp = qc.decompose()

result = qiskit.visualization.circuit_drawer(circuit_decp, output="text")
print(result)

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
result = qiskit.execute(qc, backend,shots=shots).result()
counts = result.get_counts(qc)
for res in sorted(counts.int_raw):
    print("{0:05b}".format(res))
print(counts)

