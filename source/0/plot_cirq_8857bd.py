# https://github.com/agustinsilva447/QMARL/blob/bdfd66fda61d365789ba484cf81c3ff5723a19bb/other%20files/plot_cirq.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from qiskit.circuit import Gate, Parameter
from qiskit.extensions import RXGate, RYGate 

J   = Gate(name='J', num_qubits=2, params=[])
J_d = Gate(name='J\dagger', num_qubits=2, params=[])

va1 = Parameter('\\varphi_{A1}')
va2 = Parameter('\\varphi_{A2}')
va3 = Parameter('\\varphi_{A3}')

vb1 = Parameter('\\varphi_{B1}')
vb2 = Parameter('\\varphi_{B2}')
vb3 = Parameter('\\varphi_{B3}')

circ = QuantumCircuit(2,2)
circ.append(J, [0,1])
circ.append(RXGate(va1),[0])
circ.append(RYGate(va2),[0])
circ.append(RXGate(va3),[0])
circ.append(RXGate(vb1),[1])
circ.append(RYGate(vb2),[1])
circ.append(RXGate(vb3),[1])
circ.append(J_d, [0,1])
circ.measure([0,1], [0,1])

circuit_drawer(circ, output="latex", interactive= True, scale = 2.5)