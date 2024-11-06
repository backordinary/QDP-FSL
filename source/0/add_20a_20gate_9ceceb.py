# https://github.com/harry1357/Quantum/blob/6d3b1b0970e47a86db543d82c79c98c636878b0f/add%20a%20gate.py
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
provider = IBMQ.load_account()
qc = QuantumCircuit (4,2)
qc.x(0)
qc.x(1)
qc.cx(0,2)
qc.cx(1,2)
qc.ccx(0,1,3)
qc.measure(2,0)
qc.measure(3,1)
qc.draw(output='mpl')
