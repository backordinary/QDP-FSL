# https://github.com/DGAguado/DGAVFTHackathonMadrid/blob/f36e72cb1e5c6fc6b8f60a95716fbbf6bc0835f7/levels/4qubit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

# Este programa SOBREABUSA de imágenes...
from IPython.display import Image
PATH = "encounters/"

def level3_1():
    
    qreg_q = QuantumRegister(4, 'q')
    creg_c = ClassicalRegister(4, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.barrier(qreg_q[2])
    circuit.z(qreg_q[0])
    circuit.y(qreg_q[1])
    circuit.barrier(qreg_q[3])
    circuit.h(qreg_q[0])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.y(qreg_q[3])
    circuit.h(qreg_q[3])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[3])
    circuit.id(qreg_q[2])
    
    battle=Image(filename = PATH + "encuentro4_1.png", width=300, height=300)
    
    return circuit,battle


def level3(i):
    
    if i==1:
        circuit,battle=level3_1()
        
    return circuit,battle