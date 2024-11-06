# https://github.com/agustinsilva447/QMARL/blob/bdfd66fda61d365789ba484cf81c3ff5723a19bb/paper_1/circs.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate
from qiskit.quantum_info import Operator
from qiskit.visualization import circuit_drawer
from qiskit.extensions import RXGate, RYGate, RZGate

circ = QuantumCircuit(2,2)
J = Gate(name='J(\gamma)', num_qubits=2, params=[])
J_d = Gate(name='J\dagger(\gamma)', num_qubits=2, params=[])
U4 = Gate(name='U4(\lambda)', num_qubits=1, params=[])

phia1 = Parameter('\\varphi_{A1}')
phia2 = Parameter('\\varphi_{A2}')
phia3 = Parameter('\\varphi_{A3}')
phib1 = Parameter('\\varphi_{B1}')
phib2 = Parameter('\\varphi_{B2}')
phib3 = Parameter('\\varphi_{B3}')

circ = QuantumCircuit(2,2)
circ.append(J,   [0,1])
circ.append(RXGate(phia1), [(0)])
circ.append(RYGate(phia2), [(0)])
circ.append(RXGate(phia3), [(0)])
circ.append(U4, [(0)])
circ.append(RXGate(phib1), [(1)])
circ.append(RYGate(phib2), [(1)])
circ.append(RXGate(phib3), [(1)])
circ.append(U4, [(1)])
circ.append(J_d, [0,1])
circ.measure([0,1], [0,1])

circuit_drawer(circ, output='latex', interactive = True, scale = 1.5)