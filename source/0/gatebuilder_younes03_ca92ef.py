# https://github.com/PierreEngelstein/MasterRecherche/blob/8e38d1e8825ec0778059867214c141fc04600860/Programmation/GateBuilding/GateBuilder_Younes03.py
import numpy as np
import math
import random
from qiskit import *
from qiskit.circuit.library.standard_gates import C3XGate
from qiskit.circuit.library.generalized_gates import MCMT
from qiskit.circuit.library.standard_gates.z import ZGate

def AND(circuit, a, b, control):
    circuit.h(b)
    circuit.ccx(a, b, control)

qc = QuantumCircuit(4, 4)

qc.h(0)
qc.h(1)
qc.h(2)
# qc.append(C3XGate(), [0, 1, 2, 3])
# qc.toffoli(1, 2, 3)
# qc.toffoli(0, 2, 3)
# qc.cx(2, 3)
# qc.barrier()
# qc.append(C3XGate(), [0, 1, 2, 3])
# qc.toffoli(0, 2, 3)
# qc.barrier()
# qc.append(C3XGate(), [0, 1, 2, 3])
# qc.toffoli(0, 1, 3)
# qc.barrier()
# qc.append(C3XGate(), [0, 1, 2, 3])
# qc.barrier()

qc.barrier()
# qc.append(ZGate().control(2), [1, 2, 3])
# qc.cz(2, 3)
# qc.append(ZGate().control(2), [0, 1, 3])
qc.toffoli(1, 2, 3)
qc.cx(2, 3)
qc.toffoli(0, 1, 3)

qc.barrier()
qc.x(3)
# qc.h(0)
# qc.h(1)
# qc.h(2)
qc.barrier()

qc.measure(0, 3)
qc.measure(1, 2)
qc.measure(2, 1)
qc.measure(3, 0)

result = qiskit.visualization.circuit_drawer(qc, output="text")
print(result)

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
result = qiskit.execute(qc, backend,shots=shots).result()
counts = result.get_counts(qc)
for res in sorted(counts.int_raw):
    print("{0:04b}".format(res))
print(counts)