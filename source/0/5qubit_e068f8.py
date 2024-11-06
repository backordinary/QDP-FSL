# https://github.com/DGAguado/DGAVFTHackathonMadrid/blob/f36e72cb1e5c6fc6b8f60a95716fbbf6bc0835f7/levels/5qubit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

# Este programa SOBREABUSA de imágenes...
from IPython.display import Image
PATH = "encounters/"

def level4_1():
    
    qreg_q = QuantumRegister(5, 'q')
    creg_c = ClassicalRegister(5, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.barrier(qreg_q[2])
    circuit.barrier(qreg_q[3])
    circuit.barrier(qreg_q[4])
    circuit.z(qreg_q[1])
    circuit.x(qreg_q[3])
    circuit.h(qreg_q[1])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.z(qreg_q[1])
    circuit.swap(qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[4], qreg_q[3])
    circuit.swap(qreg_q[1], qreg_q[2])
    circuit.x(qreg_q[2])
    circuit.ccx(qreg_q[3], qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[2], qreg_q[0])
    circuit.h(qreg_q[3])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.measure(qreg_q[1], creg_c[1])
    circuit.measure(qreg_q[3], creg_c[3])

    battle=Image(filename = PATH + "encuentro5_1.png", width=300, height=300)
    
    return circuit,battle

def level4_2():

    qreg_q = QuantumRegister(5, 'q')
    creg_c = ClassicalRegister(5, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.barrier(qreg_q[2])
    circuit.barrier(qreg_q[3])
    circuit.barrier(qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
    circuit.swap(qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[3])
    circuit.swap(qreg_q[0], qreg_q[1])
    circuit.ccx(qreg_q[2], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[3], qreg_q[4], qreg_q[0])
    circuit.ccx(qreg_q[4], qreg_q[0], qreg_q[1])
    circuit.measure(qreg_q[1], creg_c[1])
    
    battle=Image(filename = PATH + "encuentro5_2.png", width=300, height=300)
    
    return circuit,battle

#SKELETOR
def levelBoss_1():

    qreg_q = QuantumRegister(5, 'q')
    creg_c = ClassicalRegister(5, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[2])
    circuit.x(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[0])
    circuit.y(qreg_q[1])
    circuit.z(qreg_q[2])
    circuit.h(qreg_q[1])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.swap(qreg_q[0], qreg_q[2])
    circuit.measure(qreg_q[0], creg_c[0])
    circuit.measure(qreg_q[2], creg_c[2])
    
    battle=Image(filename = PATH + "encuentro6_1.png", width=300, height=300)
    
    return circuit,battle

def levelBoss_2():

    qreg_q = QuantumRegister(5, 'q')
    creg_c = ClassicalRegister(5, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[2])
    circuit.x(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[0])
    circuit.y(qreg_q[1])
    circuit.z(qreg_q[2])
    circuit.h(qreg_q[1])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.swap(qreg_q[0], qreg_q[2])
    circuit.measure(qreg_q[0], creg_c[0])
    circuit.measure(qreg_q[2], creg_c[2])
    
    battle=Image(filename = PATH + "encuentro6_1.png", width=300, height=300)
    
    return circuit,battle


def levelBoss_3():

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[2])
    circuit.x(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[0])
    circuit.y(qreg_q[1])
    circuit.z(qreg_q[2])
    circuit.h(qreg_q[1])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.swap(qreg_q[0], qreg_q[2])
    circuit.measure(qreg_q[0], creg_c[0])
    circuit.measure(qreg_q[2], creg_c[2])
    
    battle=Image(filename = PATH + "encuentro6_1.png", width=300, height=300)
    
    return circuit,battle


def level4(i):
    
    if i==1:
        circuit,battle=level4_1()
        
    elif i==2:
        circuit,battle=level4_2()
        
    elif i==3:
        circuit,battle=levelBoss_1()
        
    elif i==4:
        circuit,battle=levelBoss_2()
        
    elif i==5:
        circuit,battle=levelBoss_3()
        
    return circuit,battle