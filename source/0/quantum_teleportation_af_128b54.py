# https://github.com/jpineda1995/Quantum-Teleportation/blob/8e8407292063c35c7ddaddc9200220d39a958eb6/quantum_teleportation_AF.py
#!/usr/bin/env python
# coding: utf-8
#In[1]:
from numpy import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer, Aer,IBMQ
import matplotlib.pyplot as plt
import numpy as np
import math
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_bloch_multivector, plot_state_qsphere
from qiskit.extensions import Initialize
from qiskit.quantum_info import state_fidelity, partial_trace
#In[2]:
from qiskit.quantum_info import Statevector
zero = Statevector.from_label('0')
one = Statevector.from_label('1')
#In[3]:
#Number of input states
ntheta = 60
nphi = 60
dtheta = pi/(ntheta)
dphi = 2*pi/(nphi)
#Values for computation of state fidelity
sf = 0
avef = 0
for iphi in range(0,nphi):
    phi = dphi*iphi  
    for itheta in range(0,ntheta):
        theta = dtheta*itheta
            #Quantum Circuit definition (classical and quantum registers)
        qrA = QuantumRegister(1)
        qrB = QuantumRegister(2)
        qrC = QuantumRegister(1)
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(1)
        mycircuit = QuantumCircuit(qrA, qrB, qrC, cr1, cr2)
            #Input state preparation
        a = np.cos(theta/2)
        b = np.exp(1j * phi) * np.sin(theta/2)
        sv = a * zero + b * one
            #initialize qubit
        init_gate = Initialize(sv.data)
        init_gate.label = "init"
        mycircuit.append(init_gate, [0])
        mycircuit.barrier()
            #Entanglement between alice2 and bob qubits
        mycircuit.h(qrB[0])
        mycircuit.cx(qrB[0], qrB[1])
        mycircuit.barrier()
            #Bell Measurement on alice1 and alice2 qubits
        mycircuit.cx(qrA[0], qrB[0])
        mycircuit.h(qrA[0])
        mycircuit.barrier()
            #Measurement on alice's qubits
        mycircuit.measure([qrA[0],qrB[0]], [0,1])
        mycircuit.barrier()
            #Recovery operations on bob side
        mycircuit.z(qrB[1]).c_if(cr1, 1)
        mycircuit.x(qrB[1]).c_if(cr2, 1)
            #Computation of Bob state by simulation
        backend = BasicAer.get_backend('statevector_simulator')
        out_vector = execute(mycircuit, backend).result().get_statevector()
            #bob partial trace and fidelity
        new = partial_trace(out_vector, [0,1,3])
        sf = state_fidelity(sv, new)
        avef = avef + sf*sin(theta)
#Average fidelity
avet = avef/(4*pi)*dtheta*dphi
print(avet)