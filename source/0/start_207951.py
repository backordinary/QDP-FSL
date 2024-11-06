# https://github.com/tokarev-i-v/qbyq/blob/322838e4c7ce50125831afcb19e22156fce9e998/start.py
from src.circuit.circuit import QuantumCircuit
from src.state.quantum_state import QuantumState
import qiskit
from src import gates
from src.algorithms.decomposition.sc import make_cs_decompose
import numpy as np
from scipy.stats import unitary_group

operator = np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,-1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,-1,0],
    [0,0,0,0,0,0,0,1],
],)
# print(make_cs_decompose(operator, 4, 4))

qs = QuantumState(vec=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
cnot_operator = np.array([
    [1, 0, 0, 0,],
    [0, 1, 0, 0,],
    [0, 0, 0, 1,],
    [0, 0, 1, 0,],
])
gate = gates.CNOTGate(control_qubits=[0], qubits=[1])
cnotgate = gates.CNOTGate(control_qubits = [2], qubits=[3])
qc = QuantumCircuit(qubits_count=4)
qc.append(cnotgate)
qc.append(gate)

print(qs)
qc.execute(qs)
print(qc.gates_array)
print(qs)


I_op = np.array([
    [1, 0],
    [0, 1]
])
cnot_op = np.array([
    [1, 0, 0, 0,],
    [0, 1, 0, 0,],
    [0, 0, 0, 1,],
    [0, 0, 1, 0,],
])

res = np.kron(I_op, cnot_op)
res = np.kron(I_op, res)
print(res)