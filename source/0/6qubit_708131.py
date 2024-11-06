# https://github.com/DGAguado/DGAVFTHackathonMadrid/blob/f36e72cb1e5c6fc6b8f60a95716fbbf6bc0835f7/levels/6qubit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

# Este programa SOBREABUSA de imágenes...
from IPython.display import Image
PATH = "encounters/"

def level5_1():

    qreg_q = QuantumRegister(6, 'q')
    creg_c = ClassicalRegister(6, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[2])
    circuit.barrier(qreg_q[3])
    circuit.barrier(qreg_q[4])
    circuit.barrier(qreg_q[5])
    circuit.y(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.y(qreg_q[4])
    circuit.z(qreg_q[5])
    circuit.h(qreg_q[1])
    circuit.ccx(qreg_q[3], qreg_q[5], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[1])
    circuit.swap(qreg_q[1], qreg_q[4])
    circuit.swap(qreg_q[0], qreg_q[1])
    circuit.reset(qreg_q[2])
    circuit.reset(qreg_q[3])
    circuit.swap(qreg_q[4], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.h(qreg_q[4])
    circuit.h(qreg_q[5])
    circuit.cx(qreg_q[4], qreg_q[3])
    circuit.measure(qreg_q[2], creg_c[2])
    
    battle=Image(filename = PATH + "encuentroExtra.png", width=300, height=300)
    
    return circuit,battle


def level5(i):
    
    if i==1:
        circuit,battle=level5_1()
        
    return circuit,battle