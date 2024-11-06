# https://github.com/TaiteHopkins/OracleCircuit/blob/12516f1638eca0e0c2f02b1d56a955fcaa0344b9/GroverAlgorithm110101.py
#Dependency import
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import math
from qiskit.visualization import array_to_latex

#Problem Statement:
#Creating an Oracle Circuit and Diffuser which inverts the phase for the state 110101



#Defining Unitary Display function for confirmation, implementing Aer Sim

def display_unitary(qc, prefix=""):
    sim = Aer.get_backend('aer_simulator')
    qc = qc.copy()
    qc.save_unitary()
    unitary = sim.run(qc).result().get_unitary()
    display(array_to_latex(unitary, prefix=prefix))
    
#Using V-Oracle Model - Vemula et Al. 2022 arXiv:2205.00117v1 [quant-ph] 30 Apr 2022
def v_oracle(circuit):
    circuit.toffoli(0,1,6)
    circuit.toffoli(2,3,7)
    circuit.toffoli(6,7,8)
    circuit.toffoli(4,8,9)
    circuit.cz(9,5)
    circuit.toffoli(4,8,9)
    circuit.toffoli(6,7,8)
    circuit.toffoli(2,3,7)
    circuit.toffoli(0,1,6)
    
#Defining Oracle Circuit    
oracle = QuantumCircuit(10)

oracle.h([0,1,2,3,4,5])
v_oracle(oracle)
oracle.x([1,3])
oracle.x([1,3])
#Displaying Oracle phase table
display_unitary(oracle, "U_\\text{oracle}=")

#Displaying Quantum Circuit Diagram
oracle.draw()


#Diffuser Construction

diffCircuit = QuantumCircuit(10)

diffCircuit.h([0,1,2,3,4,5])
diffCircuit.x([0,1,2,3,4,5])
v_oracle(diffCircuit)
diffCircuit.x([0,1,2,3,4,5])
diffCircuit.h([0,1,2,3,4,5])


diffCircuit.draw()

grover = QuantumCircuit(10,6)
grover = grover.compose(oracle)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover = grover.compose(diffCircuit)
grover.x([1,3])

grover.measure([0,1,2,3,4,5],[0,1,2,3,4,5])
grover.draw()

sim = Aer.get_backend('aer_simulator')
counts = sim.run(grover).result().get_counts()

plot_histogram(counts)
