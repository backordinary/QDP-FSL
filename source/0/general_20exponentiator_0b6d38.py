# https://github.com/KangHaiYue/Laby-Research-Project-Quantum-computing/blob/644ed958f0905643857ec89de1c3a71d5dd27477/general%20exponentiator.py
'''general a^x where a is within 2 qubits, x within 3 qubits'''
from ibm_quantum_widgets import CircuitComposer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from math import pi

def input_translator_x(circ,x):
    '''translate x to binary and put it into the register (qubits 0-2)'''
    #x is no bigger than 7 as there is only 3 qubits to store multiplier
    x_bin = bin(x)[2:]
    x_bin_lst = [int(i) for i in x_bin]
    x_bin_lst.reverse()
    for i in range(len(x_bin_lst)):
        if x_bin_lst[i] == 1:
            circ.x(2-i)
def input_translator_a(circ,a):
    '''translate a into binary and put it into the register (qubits 3-4)'''
    #x is no bigger than 3 as there is only 2 qubits to store multiplier
    a_bin = bin(a)[2:]
    a_bin_lst = [int(i) for i in a_bin]
    a_bin_lst.reverse()
    for i in range(len(a_bin_lst)):
        if a_bin_lst[i] == 1:
            circ.x(4-i)
def output_translator(x):
    '''translate the counts from binary to decimal'''
    sum_ = list(x.keys())[0]
    sum_ = int(sum_,2)
    return sum_

def QFT(circ,start, end):
    '''quantum fourier transform specifying the index of starting and ending qubits'''
    for i in range(start,end+1):
        circ.h(i)
        for j in range(i, end+1):
            if i!=j:
                circ.cp(pi/(2**(j-i)), j, i)

def IQFT(circ,start, end):
    '''inverse quantum fourier transform specifying the index  of starting and ending qubits'''
    for i in reversed(range(start,end+1)):
        for j in reversed(range(i, end+1)):
            if i!=j:
                circ.cp(-pi/(2**(j-i)), j, i)
        circ.h(i)

def rotate2b1(circ,x1,x0,control):
    '''add 1 via phase rotation between QFT and IQFT in basis of 2 qubits'''
    circ.cp(2*pi/4,control,x1)
    circ.cp(2*pi/2,control,x0)
def un_rotate2b1(circ,x1,x0,control):
    '''minus 1 via phase rotation between QFT and IQFT in basis of 2 qubits'''
    circ.cp(-2*pi/4,control,x1)
    circ.cp(-2*pi/2,control,x0)
    
def rotate2b2(circ,x1,x0,control):
    '''add 2 via phase rotation between QFT and IQFT in basis of 2 qubits'''
    circ.cp(2*pi/2,control,x1)
def un_rotate2b2(circ,x1,x0,control):
    '''minus 2 via phase rotation between QFT and IQFT in basis of 2 qubits'''
    circ.cp(-2*pi/2,control,x1)
    
def rotate5b1(circ,x4,x3,x2,x1,x0,control):
    '''add 1 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(2*pi/32,control,x4)
    circ.cp(2*pi/16,control,x3)
    circ.cp(2*pi/8,control,x2)
    circ.cp(2*pi/4,control,x1)
    circ.cp(2*pi/2,control,x0)
def un_rotate5b1(circ,x4,x3,x2,x1,x0,control):
    '''minus 1 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(-2*pi/32,control,x4)
    circ.cp(-2*pi/16,control,x3)
    circ.cp(-2*pi/8,control,x2)
    circ.cp(-2*pi/4,control,x1)
    circ.cp(-2*pi/2,control,x0)
    
def rotate5b2(circ,x4,x3,x2,x1,x0,control):
    '''add 2 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(2*pi/16,control,x4)
    circ.cp(2*pi/8,control,x3)
    circ.cp(2*pi/4,control,x2)
    circ.cp(2*pi/2,control,x1)
def un_rotate5b2(circ,x4,x3,x2,x1,x0,control):
    '''minus 2 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(-2*pi/16,control,x4)
    circ.cp(-2*pi/8,control,x3)
    circ.cp(-2*pi/4,control,x2)
    circ.cp(-2*pi/2,control,x1)
    
def rotate5b4(circ,x4,x3,x2,x1,x0,control):
    '''add 4 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(2*pi/8,control,x4)
    circ.cp(2*pi/4,control,x3)
    circ.cp(2*pi/2,control,x2)
def un_rotate5b4(circ,x4,x3,x2,x1,x0,control):
    '''minus 4 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(-2*pi/8,control,x4)
    circ.cp(-2*pi/4,control,x3)
    circ.cp(-2*pi/2,control,x2)
    
def rotate5b8(circ,x4,x3,x2,x1,x0,control):
    '''add 8 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(2*pi/4,control,x4)
    circ.cp(2*pi/2,control,x3)
def un_rotate5b8(circ,x4,x3,x2,x1,x0,control):
    '''minus 8 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(-2*pi/4,control,x4)
    circ.cp(-2*pi/2,control,x3)
    
def rotate5b16(circ,x4,x3,x2,x1,x0,control):
    '''add 16 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(2*pi/2,control,x4)
def un_rotate5b16(circ,x4,x3,x2,x1,x0,control):
    '''minus 16 via phase rotation between QFT and IQFT in basis of 5 qubits'''
    circ.cp(-2*pi/2,control,x4)
    
def rotate7b1(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 1 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(2*pi/128,control,x6)
    circ.cp(2*pi/64,control,x5)
    circ.cp(2*pi/32,control,x4)
    circ.cp(2*pi/16,control,x3)
    circ.cp(2*pi/8,control, x2)
    circ.cp(2*pi/4,control, x1)
    circ.cp(2*pi/2,control, x0)

def un_rotate7b1(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''minus 1 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(-2*pi/128,control,x6)
    circ.cp(-2*pi/64,control,x5)
    circ.cp(-2*pi/32,control,x4)
    circ.cp(-2*pi/16,control,x3)
    circ.cp(-2*pi/8,control, x2)
    circ.cp(-2*pi/4,control, x1)
    circ.cp(-2*pi/2,control, x0)
    
def rotate7b2(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 2 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(2*pi/64,control,x6)
    circ.cp(2*pi/32,control,x5)
    circ.cp(2*pi/16,control,x4)
    circ.cp(2*pi/8,control,x3)
    circ.cp(2*pi/4,control,x2)
    circ.cp(2*pi/2,control,x1)
def un_rotate7b2(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''minus 2 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(-2*pi/64,control,x6)
    circ.cp(-2*pi/32,control,x5)
    circ.cp(-2*pi/16,control,x4)
    circ.cp(-2*pi/8,control,x3)
    circ.cp(-2*pi/4,control,x2)
    circ.cp(-2*pi/2,control,x1)
    
def rotate7b4(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 4 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(2*pi/32,control,x6)
    circ.cp(2*pi/16,control,x5)
    circ.cp(2*pi/8,control,x4)
    circ.cp(2*pi/4,control,x3)
    circ.cp(2*pi/2,control,x2)
def un_rotate7b4(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''minus 4 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(-2*pi/32,control,x6)
    circ.cp(-2*pi/16,control,x5)
    circ.cp(-2*pi/8,control,x4)
    circ.cp(-2*pi/4,control,x3)
    circ.cp(-2*pi/2,control,x2)
    
def rotate7b8(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 8 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(2*pi/16,control,x6)
    circ.cp(2*pi/8,control,x5)
    circ.cp(2*pi/4,control,x4)
    circ.cp(2*pi/2,control,x3)
def un_rotate7b8(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''minus 8 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(-2*pi/16,control,x6)
    circ.cp(-2*pi/8,control,x5)
    circ.cp(-2*pi/4,control,x4)
    circ.cp(-2*pi/2,control,x3)
    
def rotate7b16(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 16 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(2*pi/8,control,x6)
    circ.cp(2*pi/4,control,x5)
    circ.cp(2*pi/2,control,x4)
def un_rotate7b16(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''minus 16 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(-2*pi/8,control,x6)
    circ.cp(-2*pi/4,control,x5)
    circ.cp(-2*pi/2,control,x4)
    
def rotate7b32(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 32 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(2*pi/4,control,x6)
    circ.cp(2*pi/2,control,x5)
def un_rotate7b32(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''minus 32 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(-2*pi/4,control,x6)
    circ.cp(-2*pi/2,control,x5)
    
def rotate7b64(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 64 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(2*pi/2,control,x6)
def un_rotate7b64(circ,x6,x5,x4,x3,x2,x1,x0,control):
    '''minus 64 via phase rotation between QFT and IQFT in basis of 7 qubits'''
    circ.cp(-2*pi/2,control,x6)
    
def rotate4b1(circ,x3,x2,x1,x0,control):
    '''add 1 via phase rotation between QFT and IQFT in basis of 4 qubits'''
    circ.cp(2*pi/16,control,x3)
    circ.cp(2*pi/8,control, x2)
    circ.cp(2*pi/4,control, x1)
    circ.cp(2*pi/2,control, x0)
def un_rotate4b1(circ,x3,x2,x1,x0,control):
    '''minus 1 via phase rotation between QFT and IQFT in basis of 4 qubits'''
    circ.cp(-2*pi/16,control,x3)
    circ.cp(-2*pi/8,control, x2)
    circ.cp(-2*pi/4,control, x1)
    circ.cp(-2*pi/2,control, x0)
    
def rotate4b2(circ,x3,x2,x1,x0,control):
    '''add 2 via phase rotation between QFT and IQFT in basis of 4 qubits'''
    circ.cp(2*pi/8,control,x3)
    circ.cp(2*pi/4,control,x2)
    circ.cp(2*pi/2,control,x1)
def un_rotate4b2(circ,x3,x2,x1,x0,control):
    '''minus 2 via phase rotation between QFT and IQFT in basis of 4 qubits'''
    circ.cp(-2*pi/8,control,x3)
    circ.cp(-2*pi/4,control,x2)
    circ.cp(-2*pi/2,control,x1)
    
def rotate4b4(circ,x3,x2,x1,x0,control):
    '''add 4 via phase rotation between QFT and IQFT in basis of 4 qubits'''
    circ.cp(2*pi/4,control,x3)
    circ.cp(2*pi/2,control,x2)
def un_rotate4b4(circ,x3,x2,x1,x0,control):
    '''minus 4 via phase rotation between QFT and IQFT in basis of 4 qubits'''
    circ.cp(-2*pi/4,control,x3)
    circ.cp(-2*pi/2,control,x2)
    
def rotate12b1(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 1 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/4096,control,x11)
    circ.cp(2*pi/2048,control,x10)
    circ.cp(2*pi/1024,control,x9)
    circ.cp(2*pi/512,control,x8)
    circ.cp(2*pi/256,control,x7)
    circ.cp(2*pi/128,control,x6)
    circ.cp(2*pi/64,control,x5)
    circ.cp(2*pi/32,control,x4)
    circ.cp(2*pi/16,control,x3)
    circ.cp(2*pi/8,control, x2)
    circ.cp(2*pi/4,control, x1)
    circ.cp(2*pi/2,control, x0)

def rotate12b2(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 2 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/2048,control,x11)
    circ.cp(2*pi/1024,control,x10)
    circ.cp(2*pi/512,control,x9)
    circ.cp(2*pi/256,control,x8)
    circ.cp(2*pi/128,control,x7)
    circ.cp(2*pi/64,control,x6)
    circ.cp(2*pi/32,control,x5)
    circ.cp(2*pi/16,control,x4)
    circ.cp(2*pi/8,control, x3)
    circ.cp(2*pi/4,control, x2)
    circ.cp(2*pi/2,control, x1)
    # note beacuse last rotation is 0 degree so no need to apply controlled phase anymore

def rotate12b4(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 4 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/1024,control,x11)
    circ.cp(2*pi/512,control,x10)
    circ.cp(2*pi/256,control,x9)
    circ.cp(2*pi/128,control,x8)
    circ.cp(2*pi/64,control,x7)
    circ.cp(2*pi/32,control,x6)
    circ.cp(2*pi/16,control,x5)
    circ.cp(2*pi/8,control, x4)
    circ.cp(2*pi/4,control, x3)
    circ.cp(2*pi/2,control, x2)
    # note beacuse last 2 rotations is 0 degree so no need to apply controlled phase anymore

def rotate12b8(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 8 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/512,control,x11)
    circ.cp(2*pi/256,control,x10)
    circ.cp(2*pi/128,control,x9)
    circ.cp(2*pi/64,control,x8)
    circ.cp(2*pi/32,control,x7)
    circ.cp(2*pi/16,control,x6)
    circ.cp(2*pi/8,control, x5)
    circ.cp(2*pi/4,control, x4)
    circ.cp(2*pi/2,control, x3)
    # last 3 rotations are 0
def rotate12b16(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 16 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/256,control,x11)
    circ.cp(2*pi/128,control,x10)
    circ.cp(2*pi/64,control,x9)
    circ.cp(2*pi/32,control,x8)
    circ.cp(2*pi/16,control,x7)
    circ.cp(2*pi/8,control, x6)
    circ.cp(2*pi/4,control, x5)
    circ.cp(2*pi/2,control, x4)
    # last 4 rotations are 0
    
def rotate12b32(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 32 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/128,control,x11)
    circ.cp(2*pi/64,control,x10)
    circ.cp(2*pi/32,control,x9)
    circ.cp(2*pi/16,control,x8)
    circ.cp(2*pi/8,control, x7)
    circ.cp(2*pi/4,control, x6)
    circ.cp(2*pi/2,control, x5)
    # last 5 rotations are 0

def rotate12b64(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 64 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/64,control,x11)
    circ.cp(2*pi/32,control,x10)
    circ.cp(2*pi/16,control,x9)
    circ.cp(2*pi/8,control, x8)
    circ.cp(2*pi/4,control, x7)
    circ.cp(2*pi/2,control, x6)
    # last 6 rotations are 0

def rotate12b128(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 128 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/32,control,x11)
    circ.cp(2*pi/16,control,x10)
    circ.cp(2*pi/8,control, x9)
    circ.cp(2*pi/4,control, x8)
    circ.cp(2*pi/2,control, x7)
    # last 6 rotations are 0
    
def rotate12b256(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 256 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/16,control,x11)
    circ.cp(2*pi/8,control, x10)
    circ.cp(2*pi/4,control, x9)
    circ.cp(2*pi/2,control, x8)
    # last 6 rotations are 0

def rotate12b512(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 512 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/8,control, x11)
    circ.cp(2*pi/4,control, x10)
    circ.cp(2*pi/2,control, x9)
    # last 6 rotations are 0

def rotate12b1024(circ,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,x0,control):
    '''add 1024 via phase rotation between QFT and IQFT in basis of 12 qubits'''
    circ.cp(2*pi/4,control, x11)
    circ.cp(2*pi/2,control, x10)
    # last 6 rotations are 0

def exponentiator(a,x):
    '''computes a^x, where a within 2 qubits, x within 3 qubits'''
    circuit = QuantumCircuit(42,12)#initialize the circuit
    #0-2 for register x, 3-4 for register a, 5-17 for calculating a^2, a^4 and 12,13 are
    #common control qubits, the rests are accumulator
    input_translator_x(circuit,x)#register x
    input_translator_a(circuit,a)#register a
    circuit.x(37)#initialize the accumulator to be 1
    
    #multiply accumulator by a if first digit of x is 1, otherwise keep it
    QFT(circuit,30,31)
    circuit.toffoli(4,37,13)
    circuit.toffoli(2,13,12)
    rotate2b1(circuit,30,31,12)
    circuit.toffoli(2,13,12)
    circuit.toffoli(4,37,13)
    
    circuit.toffoli(3,37,13)
    circuit.toffoli(2,13,12)
    rotate2b2(circuit,30,31,12)
    circuit.toffoli(2,13,12)
    circuit.toffoli(3,37,13)
    IQFT(circuit,30,31)
    circuit.x(2)
    circuit.cx(2,31)
    circuit.x(2)
    
    #calculate a^2
    circuit.cx(3,5)
    circuit.cx(4,6)
    QFT(circuit,14,17)
    circuit.toffoli(4,6-0,13-0)
    circuit.toffoli(4,6-1,13-1)
    rotate4b1(circuit,14,15,16,17,13)
    rotate4b2(circuit,14,15,16,17,12)
    circuit.toffoli(4,6-0,13-0)
    circuit.toffoli(4,6-1,13-1)
    circuit.toffoli(3,6-0,13-0)
    circuit.toffoli(3,6-1,13-1)
    rotate4b2(circuit,14,15,16,17,13)
    rotate4b4(circuit,14,15,16,17,12)
    circuit.toffoli(3,6-0,13-0)
    circuit.toffoli(3,6-1,13-1)
    IQFT(circuit,14,17)
    circuit.cx(3,5)
    circuit.cx(4,6)
    
    #multiply accumulator by a^2 if 2nd digit of x is 1, otherwise keep it
    QFT(circuit,32,36)
    circuit.toffoli(31,17-0,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**0))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-0,13)
    circuit.toffoli(31,17-1,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**1))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-1,13)
    circuit.toffoli(31,17-2,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**2))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-2,13)
    circuit.toffoli(31,17-3,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**3))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-3,13)
    circuit.toffoli(30,17-0,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**(0+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-0,13)
    circuit.toffoli(30,17-1,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**(1+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-1,13)
    circuit.toffoli(30,17-2,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**(2+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-2,13)
    circuit.toffoli(30,17-3,13)
    circuit.toffoli(1,13,12)
    eval('rotate5b{}'.format(2**(3+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-3,13)
    IQFT(circuit,32,36)
    circuit.x(1)
    circuit.toffoli(1,31,36)
    circuit.toffoli(1,30,35)
    circuit.x(1)
    circuit.cx(14+0,18+0)
    circuit.cx(14+1,18+1)
    circuit.cx(14+2,18+2)
    circuit.cx(14+3,18+3)
    QFT(circuit,5,11)
    circuit.toffoli(21-0,17,13)
    eval('rotate7b{}'.format(2**0))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-0,17,13)
    circuit.toffoli(21-1,17,13)
    eval('rotate7b{}'.format(2**1))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-1,17,13)
    circuit.toffoli(21-2,17,13)
    eval('rotate7b{}'.format(2**2))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-2,17,13)
    circuit.toffoli(21-3,17,13)
    eval('rotate7b{}'.format(2**3))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-3,17,13)
    circuit.toffoli(21-0,16,13)
    eval('rotate7b{}'.format(2**(0+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-0,16,13) 
    circuit.toffoli(21-1,16,13)
    eval('rotate7b{}'.format(2**(1+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-1,16,13) 
    circuit.toffoli(21-2,16,13)
    eval('rotate7b{}'.format(2**(2+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-2,16,13) 
    circuit.toffoli(21-3,16,13)
    eval('rotate7b{}'.format(2**(3+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-3,16,13) 
    circuit.toffoli(21-0,15,13)
    eval('rotate7b{}'.format(2**(0+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-0,15,13)
    circuit.toffoli(21-1,15,13)
    eval('rotate7b{}'.format(2**(1+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-1,15,13)
    circuit.toffoli(21-2,15,13)
    eval('rotate7b{}'.format(2**(2+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-2,15,13)
    circuit.toffoli(21-3,15,13)
    eval('rotate7b{}'.format(2**(3+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-3,15,13)
    circuit.toffoli(21-0,14,13)
    eval('rotate7b{}'.format(2**(0+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-0,14,13)
    circuit.toffoli(21-1,14,13)
    eval('rotate7b{}'.format(2**(1+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-1,14,13)
    circuit.toffoli(21-2,14,13)
    eval('rotate7b{}'.format(2**(2+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-2,14,13)
    circuit.toffoli(21-3,14,13)
    eval('rotate7b{}'.format(2**(3+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(21-3,14,13)
    IQFT(circuit,5,11)
    circuit.cx(14+0,18+0)
    circuit.cx(14+1,18+1)
    circuit.cx(14+2,18+2)
    circuit.cx(14+3,18+3)

    #multiply accumulator by a^4 if 3rd digit of x is 1, otherwise keep it
    QFT(circuit,18,29)
    circuit.toffoli(11,36-0,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**0))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(11,36-0,13)
    circuit.toffoli(11,36-1,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**1))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(11,36-1,13)
    circuit.toffoli(11,36-2,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**2))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(11,36-2,13)
    circuit.toffoli(11,36-3,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**3))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(11,36-3,13)
    circuit.toffoli(11,36-4,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**4))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(11,36-4,13)
    circuit.toffoli(10,36-0,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(0+1)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(10,36-0,13)
    circuit.toffoli(10,36-1,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(1+1)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(10,36-1,13)
    circuit.toffoli(10,36-2,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(2+1)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(10,36-2,13)
    circuit.toffoli(10,36-3,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(3+1)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(10,36-3,13)
    circuit.toffoli(10,36-4,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(4+1)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(10,36-4,13)
    circuit.toffoli(9,36-0,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(0+2)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(9,36-0,13)
    circuit.toffoli(9,36-1,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(1+2)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(9,36-1,13)
    circuit.toffoli(9,36-2,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(2+2)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(9,36-2,13)
    circuit.toffoli(9,36-3,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(3+2)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(9,36-3,13)
    circuit.toffoli(9,36-4,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(4+2)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(9,36-4,13)
    circuit.toffoli(8,36-0,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(0+3)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(8,36-0,13)
    circuit.toffoli(8,36-1,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(1+3)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(8,36-1,13)
    circuit.toffoli(8,36-2,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(2+3)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(8,36-2,13)
    circuit.toffoli(8,36-3,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(3+3)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(8,36-3,13)
    circuit.toffoli(8,36-4,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(4+3)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(8,36-4,13)
    circuit.toffoli(7,36-0,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(0+4)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(7,36-0,13)
    circuit.toffoli(7,36-1,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(1+4)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(7,36-1,13)
    circuit.toffoli(7,36-2,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(2+4)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(7,36-2,13)
    circuit.toffoli(7,36-3,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(3+4)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(7,36-3,13)
    circuit.toffoli(7,36-4,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(4+4)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(7,36-4,13)
    circuit.toffoli(6,36-0,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(0+5)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(6,36-0,13)
    circuit.toffoli(6,36-1,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(1+5)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(6,36-1,13)
    circuit.toffoli(6,36-2,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(2+5)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(6,36-2,13)
    circuit.toffoli(6,36-3,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(3+5)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(6,36-3,13)
    circuit.toffoli(6,36-4,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(4+5)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(6,36-4,13)
    circuit.toffoli(5,36-0,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(0+6)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(5,36-0,13)
    circuit.toffoli(5,36-1,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(1+6)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(5,36-1,13)
    circuit.toffoli(5,36-2,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(2+6)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(5,36-2,13)
    circuit.toffoli(5,36-3,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(3+6)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(5,36-3,13)
    circuit.toffoli(5,36-4,13)
    circuit.toffoli(0,13,12)
    eval('rotate12b{}'.format(2**(4+6)))(circuit,18,19,20,21,22,23,24,25,26,27,28,29,12)
    circuit.toffoli(0,13,12)
    circuit.toffoli(5,36-4,13)
    IQFT(circuit,18,29)
    circuit.x(0)
    circuit.toffoli(0,36-0,29-0)
    circuit.toffoli(0,36-1,29-1)
    circuit.toffoli(0,36-2,29-2)
    circuit.toffoli(0,36-3,29-3)
    circuit.toffoli(0,36-4,29-4)
    circuit.x(0)
    circuit.cx(14+0,38+0)
    circuit.cx(14+1,38+1)
    circuit.cx(14+2,38+2)
    circuit.cx(14+3,38+3)
    QFT(circuit,5,11)
    circuit.toffoli(41-0,17,13)
    eval('un_rotate7b{}'.format(2**0))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-0,17,13)
    circuit.toffoli(41-1,17,13)
    eval('un_rotate7b{}'.format(2**1))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-1,17,13)
    circuit.toffoli(41-2,17,13)
    eval('un_rotate7b{}'.format(2**2))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-2,17,13)
    circuit.toffoli(41-3,17,13)
    eval('un_rotate7b{}'.format(2**3))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-3,17,13)
    circuit.toffoli(41-0,16,13)
    eval('un_rotate7b{}'.format(2**(0+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-0,16,13) 
    circuit.toffoli(41-1,16,13)
    eval('un_rotate7b{}'.format(2**(1+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-1,16,13) 
    circuit.toffoli(41-2,16,13)
    eval('un_rotate7b{}'.format(2**(2+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-2,16,13) 
    circuit.toffoli(41-3,16,13)
    eval('un_rotate7b{}'.format(2**(3+1)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-3,16,13) 
    circuit.toffoli(41-0,15,13)
    eval('un_rotate7b{}'.format(2**(0+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-0,15,13)
    circuit.toffoli(41-1,15,13)
    eval('un_rotate7b{}'.format(2**(1+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-1,15,13)
    circuit.toffoli(41-2,15,13)
    eval('un_rotate7b{}'.format(2**(2+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-2,15,13)
    circuit.toffoli(41-3,15,13)
    eval('un_rotate7b{}'.format(2**(3+2)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-3,15,13)
    circuit.toffoli(41-0,14,13)
    eval('un_rotate7b{}'.format(2**(0+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-0,14,13)
    circuit.toffoli(41-1,14,13)
    eval('un_rotate7b{}'.format(2**(1+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-1,14,13)
    circuit.toffoli(41-2,14,13)
    eval('un_rotate7b{}'.format(2**(2+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-2,14,13)
    circuit.toffoli(41-3,14,13)
    eval('un_rotate7b{}'.format(2**(3+3)))(circuit,5,6,7,8,9,10,11,13)
    circuit.toffoli(41-3,14,13)
    IQFT(circuit,5,11)
    circuit.cx(14+0,38+0)
    circuit.cx(14+1,38+1)
    circuit.cx(14+2,38+2)
    circuit.cx(14+3,38+3)
    #undo working qubits multipled a^2
    circuit.x(1)
    circuit.toffoli(1,31,36)
    circuit.toffoli(1,30,35)
    circuit.x(1)  
    QFT(circuit,32,36)
    circuit.toffoli(31,17-0,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**0))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-0,13)
    circuit.toffoli(31,17-1,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**1))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-1,13)
    circuit.toffoli(31,17-2,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**2))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-2,13)
    circuit.toffoli(31,17-3,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**3))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(31,17-3,13)
    circuit.toffoli(30,17-0,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**(0+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-0,13)
    circuit.toffoli(30,17-1,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**(1+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-1,13)
    circuit.toffoli(30,17-2,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**(2+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-2,13)
    circuit.toffoli(30,17-3,13)
    circuit.toffoli(1,13,12)
    eval('un_rotate5b{}'.format(2**(3+1)))(circuit,32,33,34,35,36,12)
    circuit.toffoli(1,13,12)
    circuit.toffoli(30,17-3,13)
    IQFT(circuit,32,36)
  
    
    
    #undo working qubits calculated a^2
    circuit.cx(3,5)
    circuit.cx(4,6)
    QFT(circuit,14,17)
    circuit.toffoli(4,6-0,13-0)
    circuit.toffoli(4,6-1,13-1)
    un_rotate4b1(circuit,14,15,16,17,13)
    un_rotate4b2(circuit,14,15,16,17,12)
    circuit.toffoli(4,6-0,13-0)
    circuit.toffoli(4,6-1,13-1)
    circuit.toffoli(3,6-0,13-0)
    circuit.toffoli(3,6-1,13-1)
    un_rotate4b2(circuit,14,15,16,17,13)
    un_rotate4b4(circuit,14,15,16,17,12)
    circuit.toffoli(3,6-0,13-0)
    circuit.toffoli(3,6-1,13-1)
    IQFT(circuit,14,17)
    circuit.cx(3,5)
    circuit.cx(4,6)
    
    #undo working qubits multplied a
    circuit.x(2)
    circuit.cx(2,31)
    circuit.x(2)
    QFT(circuit,30,31)
    circuit.toffoli(4,37,13)
    circuit.toffoli(2,13,12)
    un_rotate2b1(circuit,30,31,12)
    circuit.toffoli(2,13,12)
    circuit.toffoli(4,37,13)
    
    circuit.toffoli(3,37,13)
    circuit.toffoli(2,13,12)
    un_rotate2b2(circuit,30,31,12)
    circuit.toffoli(2,13,12)
    circuit.toffoli(3,37,13)
    IQFT(circuit,30,31)
    
    #undo initial register |1>
    circuit.x(37)
    
    # measure the final result a^x
    for i in range(12):
        circuit.measure(29-i,0+i)
    #put the circuit into simulator and run it
    backend = QasmSimulator()
    qc_compiled = transpile(circuit, backend)
    job = backend.run(qc_compiled, shots = 1)
    result = job.result()
    counts = result.get_counts(qc_compiled)
    return output_translator(counts)
