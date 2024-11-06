# https://github.com/qismib/QuITE/blob/ca89443c69dc3d2e06fee4a2339a6e52cd8fcb81/programs/real_device/functions_real_device.py
from qiskit import *
from qiskit.extensions import Initialize
from qiskit.visualization import *
from qiskit.quantum_info import *
import numpy as np
import random
import math

#angoli casuali per creare psi
a = random.random()
b = random.random()

theta = a*2*np.pi
phi = b*np.pi



def Bell_pair(circuit, qr) :
    circuit.h(qr[1])
    circuit.cx(qr[1],qr[2])

    circuit.barrier()

#non utilizzata
def create_a_random_state0(circuit, qr) :
    vec = random_statevector(2) #creo un generico vettore a base 2
    #al qubit 0 do le ampiezze del generico vettore
    circuit.initialize(vec.data, qr[0])

    plot_bloch_multivector(vec)

    circuit.barrier()


#utilizzata
def create_a_random_state1(circuit, qr) :
    #la trasformazione che porta da |0> a |psi> può essere sempre riscritta cosi
    circuit.ry(phi, qr[0])
    circuit.rz(theta, qr[0])

    plot_bloch_vector([1,phi,theta], title='psi', coord_type='spherical')

    circuit.barrier()


def Alice_operations(circuit, qr) :
    circuit.cx(qr[0],qr[1])
    circuit.h(qr[0])

    circuit.barrier()

def Alice_measurements(circuit, qr, Alice_cr0, Alice_cr1) :
    circuit.measure(qr[0], Alice_cr0)
    circuit.measure(qr[1], Alice_cr1)


def Bob_operations(circuit, qr, Alice_cr0, Alice_cr1) :
     circuit.x(qr[2]).c_if(Alice_cr1, 1)
     circuit.z(qr[2]).c_if(Alice_cr0, 1)

     circuit.barrier()

def Bob_operations_real_device(circuit, qr) :
    circuit.cx(qr[1], qr[2])
    circuit.cz(qr[0], qr[2])

    circuit.barrier()


def checking(circuit, qr, Bob_cr) :
    #riporto lo stato a |0>
    circuit.rz(2*np.pi-theta, qr[2])
    circuit.ry(2*np.pi-phi, qr[2])

    circuit.barrier()
