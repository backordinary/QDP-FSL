# https://github.com/3gaspo/guide-infoQ/blob/ae8ec94a5bfb715168017518abb4beb51c969713/codes/operations_de_base.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

#opérateurs 1 bit
qc = QuantumCircuit(1)
qc.x(0) #Porte NOT
qc.h(0) #Porte Hadamard
qc.y(0) #Porte Y
qc.z(0) #Porte Z
qc.draw()


#opérateurs 2 bits
qc = QuantumCircuit(2)
qc.swap(0,1) #Porte SWAP
qc.cx(0,1) #Porte CNOT, bit de contrôle : 0
qc.draw()


#comparaison de circuits
from qiskit.quantum_info import Statevector
def compare(qc1,qc2):
    #compare si deux circuits sont équivalents
    return Statevector.from_instruction(qc1).equiv(Statevector.from_instruction(qc2))

qc1 = QuantumCircuit(1,1)
qc2 = QuantumCircuit(1,1)

qc1.z(0)
qc2.h(0)
qc2.x(0)
qc2.h(0)

compare(qc1,qc2)

qc1 = QuantumCircuit(1,1)
qc2 = QuantumCircuit(1,1)

qc1.x(0)
qc2.h(0)
qc2.z(0)
qc2.h(0)

compare(qc1,qc2)