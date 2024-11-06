# https://github.com/DGAguado/DGAVFTHackathonMadrid/blob/f36e72cb1e5c6fc6b8f60a95716fbbf6bc0835f7/levels/3qubit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

# Este programa SOBREABUSA de imágenes...
from IPython.display import Image
PATH = "encounters/"

def level2_1():

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.barrier(qreg_q[2])
    circuit.y(qreg_q[0])
    circuit.ccx(qreg_q[2], qreg_q[0], qreg_q[1])
    circuit.h(qreg_q[0])
    circuit.measure(qreg_q[1], creg_c[1])
    
    battle=Image(filename = PATH + "encuentro3_1.png", width=300, height=300)
    
    return circuit,battle

def level2_2():

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.barrier(qreg_q[2])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[0])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.measure(qreg_q[0], creg_c[0])
    
    battle=Image(filename = PATH + "encuentro3_2.png", width=300, height=300)
    
    return circuit,battle

def level2_3():

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.h(qreg_q[1])
    circuit.barrier(qreg_q[2])
    circuit.swap(qreg_q[1], qreg_q[2])
    circuit.z(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.swap(qreg_q[0], qreg_q[1])
    circuit.h(qreg_q[0])
    circuit.measure(qreg_q[1], creg_c[1])
    
    battle=Image(filename = PATH + "encuentro3_3.png", width=300, height=300)
    
    return circuit,battle

def level2_4():

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.barrier(qreg_q[2])
    circuit.y(qreg_q[2])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[1])
    circuit.z(qreg_q[2])
    circuit.measure(qreg_q[1], creg_c[1])
    circuit.measure(qreg_q[2], creg_c[2])

    battle=Image(filename = PATH + "encuentro3_4.png", width=300, height=300)
    
    return circuit,battle
    
def level2_5():

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[0])
    circuit.barrier(qreg_q[2])
    circuit.x(qreg_q[0])
    circuit.barrier(qreg_q[1])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
    circuit.swap(qreg_q[0], qreg_q[1])
    circuit.swap(qreg_q[1], qreg_q[2])
    circuit.y(qreg_q[1])
    circuit.ccx(qreg_q[2], qreg_q[0], qreg_q[1])
    circuit.measure(qreg_q[1], creg_c[1])
    
    battle=Image(filename = PATH + "encuentro3_5.png", width=300, height=300)
    
    return circuit,battle


def level2(i):
    
    if i==1:
        circuit,battle=level2_1()
        
    elif i==2:
        circuit,battle=level2_2()
        
    elif i==3:
        circuit,battle=level2_3()
        
    elif i==4:
        circuit,battle=level2_4()
        
    elif i==5:
        circuit,battle=level2_5()
        
    return circuit,battle