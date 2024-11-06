# https://github.com/iQuHACK/2021_quanta/blob/964b24884b685cf374de2ddf377992f89996dd48/basic_ops.py
# Basic operations on qubits for Tiq-Taq-Toe
#import Aer here, before calling qiskit_ionq_provider
from qiskit import Aer, execute
# import all the other stuff
from qiskit_ionq_provider import IonQProvider
from qiskit import QuantumCircuit
import math

with open(".ionqkey.txt") as keyfile:
    key = keyfile.readline().strip('\n')
#Call provider and set token value
provider = IonQProvider(token=key)#Call provider and set token value

# initialize board with side length n (currently allows 1 (trivial), 2 and 3). Adds an ancilla qubit for each team. Each qubit is associated with a classical channel for readout.
def initialize(n):
    circ = QuantumCircuit(n*n,n*n)
    circ.rx(math.pi/2.,range(n*n))
    return circ

# entangle the team qubit with a chosen location on the board (numbered starting at 1 from the upper left corner and proceeding left to right by row.)
def add_cnot(circ,control,target):
    circ.cx(control,target)
    return circ

#rotation around x axis
def add_rx(circ, direction, loc):
	if direction==1:
		theta = math.pi/4.
	else: theta = -math.pi/4.
	circ.rx(theta, loc)
	return circ

#rotation around y axis
def add_ry(circ, direction, loc):
	if direction==1:
		theta = math.pi/4.
	else: theta = -math.pi/4.
	circ.ry(theta, loc)
	return circ

#measure final state, n is the side length of the board
def add_measurement(circ,n):
	circ.measure(range(n*n),range(n*n))
	return circ
