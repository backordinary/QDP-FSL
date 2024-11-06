# https://github.com/PierreEngelstein/MasterRecherche/blob/8e38d1e8825ec0778059867214c141fc04600860/Programmation/GateBuilding/AND.py
import numpy as np
import math
import random
from qiskit import *

def AND():
    '''
       Basic AND circuit using Toffoli Gate
    '''
    _circ = QuantumCircuit(3, name="AND")
    _circ.reset(2)
    _circ.ccx(0, 1, 2)
    return _circ

def NOT():
    _circ = QuantumCircuit(1, name="NOT")
    _circ.x(0)
    return _circ

qc = QuantumCircuit(3, 3)
qr = qc.qregs[0]
qc.h(0)
qc.h(1)
qc.h(2)

qc.barrier()
qc.append(AND().to_instruction(), [qr[0], qr[1], qr[2]])
# qc.barrier()
# qc.append(AND().to_instruction(), [qr[2], qr[3], qr[4]])
# qc.barrier()
# qc.append(AND().to_instruction(), [qr[4], qr[5], qr[2]])
# qc.append(NOT().to_instruction(), [qr[5], qr[6]])
qc.barrier()
qc.measure(0, 0)
qc.measure(1, 1)
qc.measure(2, 2)
# qc.measure(5, 3)
# qc.measure(2, 4)

result = qiskit.visualization.circuit_drawer(qc, output="text")
print(result)

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
result = qiskit.execute(qc, backend,shots=shots).result()
counts = result.get_counts(qc)
for res in sorted(counts.int_raw):
    print("{0:05b}".format(res), counts.int_raw[res])
