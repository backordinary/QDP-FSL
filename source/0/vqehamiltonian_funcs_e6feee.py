# https://github.com/EeshGupta/VQE_Research/blob/98a0674eca203119b0c9ae358ce0379ba0a869fa/Coding%20Runs/Experiments/November/Schnetzer%20Readout%20Method/VQEHamiltonian_funcs.py
import math
from qiskit import ClassicalRegister

def circs(circuits):
    '''
    Gives copy of list of circuits
    '''
    n_circuits = [circ.copy() for circ in circuits ]
    return n_circuits

def Hammifier(circy, Hamiltonian):
    """
    Adding local hamiltonians and returning one for meas II, IZ . ZI. ZZ and other for XX
    """
    circuits = []
    for line in iter(Hamiltonian.splitlines()):
        circ = circy.copy()
        for index in range(len(line)):
            op = line[index]

            ##do nothing if Z or I
            if(op == "X"): 
                #hadamard in u3
                circ.u3(math.pi/2, 0, math.pi, index)
            elif(op == "Y"):
                circ.rx(pi/2, index)
        #c = ClassicalRegister(2)
        #circ.add_register(c)
        #circ.measure([0], [0])
        #circ.measure([1],[1])
        circuits.append(circ)
    return [circuits[0], circuits[4]]