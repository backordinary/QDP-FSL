# https://github.com/kakipi76/AQUA/blob/4d1da3ac90f68efd47ac32025d2577824d4ccefe/Quantum_Walk.py
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.circuit.library import QFT
from numpy import pi
from qiskit.quantum_info import Statevector
from matplotlib import pyplot as plt
import numpy as np
# Loading your IBM Q account(s)
provider = IBMQ.load_account()

one_step_circuit = QuantumCircuit(6, name=' ONE STEP') 
# Coin operator
one_step_circuit.h([4,5])
one_step_circuit.z([4,5])
one_step_circuit.cz(4,5)
one_step_circuit.h([4,5])
one_step_circuit.draw() 

# Shift operator function for 4d-hypercube
def shift_operator(circuit):
    circuit.x(4)
    if 0%2==0:
        circuit.x(5)
    circuit.ccx(4,5,0)
    circuit.x(4)
    if 1%2==0:
        circuit.x(5)
    circuit.ccx(4,5,1)
    circuit.x(4)
    if 2%2==0:
        circuit.x(5)
    circuit.ccx(4,5,2)
    circuit.x(4)
    if 3%2==0:
        circuit.x(5)
    circuit.ccx(4,5,3)

shift_operator(one_step_circuit)

one_step_gate = one_step_circuit.to_instruction() 
one_step_circuit.draw()

backend = Aer.get_backend('qasm_simulator') 
job = execute( one_step_circuit, backend, shots=1024 ) 
hist = job.result().get_counts() 
plot_histogram( hist )
