# https://github.com/amaya-the-grey/Old_Codes/blob/59169c4eff9bcebfa7f07b308b72eec01613219b/3coloring.py
#initialization
#import matplotlib.pyplot as plt
##%matplotlib inline
import numpy as np

# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

# import basic plot tools
#from qiskit.tools.visualization import plot_histogram

qr = QuantumRegister(19)
cr = ClassicalRegister(3)
circuit = QuantumCircuit(qr,cr)
#phase_oracle(oracleCircuit, qr)
#oracleCircuit.draw(output="mpl")

# initializing all three vertices joined
circuit.x(qr[0])
circuit.x(qr[2])
circuit.x(qr[4])

# checking edges, expect |111> output
circuit.cx(0,6)
circuit.cx(1,6)
circuit.cx(2,7)
circuit.cx(3,7)
circuit.cx(4,8)
circuit.cx(5,8)

# hardwired 3-color logic (maybe?)
# first gate
circuit.ccx(6,7,11)
circuit.ccx(11,8,18)
# second gate
circuit.cx(6,9)
circuit.ccx(9,7,18)
# third gate
circuit.cx(7,10)
circuit.ccx(6,10,17)
# fourth gate
circuit.ccx(6,7,12)
circuit.cx(8,13)
circuit.ccx(12,13,16)
# fifth gate
circuit.ccx(6,7,14)
circuit.ccx(8,14,15)
circuit.cx(15,17)
circuit.ch(15,18)

# measure
circuit.measure(qr[16:],cr)


# hybrid logic
#print(qr[0:6])
#if (qr[6] == 1):
#	print("Hi")


#def n_controlled_Z(circuit, controls, target):
#    """Implement a Z gate with multiple controls"""
#    if (len(controls) > 2):
#        raise ValueError('The controlled Z with more than 2 controls is not implemented')
#    elif (len(controls) == 1):
#        circuit.h(target)
#        circuit.cx(controls[0], target)
#        circuit.h(target)
#    elif (len(controls) == 2):
#        circuit.h(target)
#        circuit.ccx(controls[0], controls[1], target)
#        circuit.h(target)


backend = BasicAer.get_backend('qasm_simulator')
result = execute(circuit, backend=backend, shots=1000).result()
answer = result.get_counts()
print(answer)
