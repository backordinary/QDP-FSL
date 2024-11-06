# https://github.com/PierreEngelstein/MasterRecherche/blob/8e38d1e8825ec0778059867214c141fc04600860/Programmation/GateBuilding/function_evaluate_dj.py
from qiskit import *  
from qiskit.circuit.library.standard_gates.z import ZGate

def AND(circ, a, b):
    circ.append(ZGate().control(1), [a, b])

def OR(circ, a, b):
    circ.z(a)
    circ.z(b)
    circ.append(ZGate().control(1), [a, b])

qc = QuantumCircuit(3, 3)
# qc.h(0)
# qc.h(1)
# qc.h(2)

# qc.x(1)
# qc.z(1)
# qc.ccz(0, 1, 2)
# AND(qc, 0, 1)

# qc.z(0)
# qc.x(1)
# qc.x(0)
qc.toffoli(0, 1, 2)
qc.cx(1, 2)
qc.cx(0, 2)
qc.x(2)
# OR(qc, 1, 2)
# AND(qc, 2, 0)
# qc.append(ZGate().control(1), [1, 0])

target = qiskit.quantum_info.Statevector.from_instruction(qc)
for i in target.data:
    print("{:.04f}".format(float(i.real)))

print("**********")

result = qiskit.visualization.circuit_drawer(qc, output="text")
print(result)

print("**********")

import numpy as np
backend = Aer.get_backend('unitary_simulator')
job = execute(qc, backend)
result = job.result()
result_matrix = result.get_unitary(qc, decimals=3)
a = np.asarray(result_matrix)
for i in a:
    for j in i:
        if int(j.real) == 0:
            print(int(j.real), end= '  ')
        elif int(j.real) == -1:
            print(int(j.real), end= ' ')
        else:
            print(int(j.real), end= '  ')
    print()