# https://github.com/jpineda1995/Quantum-Teleportation-on-IBM-s-quantum-computers/blob/293825284bd567e85740281edfcc10c6a352f8fd/quantum_teleportation_RD.py
#!/usr/bin/env python
# coding: utf-8
#In[1]:
#Needed packages
from numpy import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile,execute, BasicAer, Aer,IBMQ
import matplotlib.pyplot as plt
import numpy as np
import math
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_bloch_multivector, plot_state_qsphere
from qiskit.extensions import Initialize
from qiskit.quantum_info import state_fidelity, partial_trace, Statevector
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
#In[2]:
zero = Statevector.from_label('0')
one = Statevector.from_label('1')
#In[3]:
#Choosing the desired device
IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q')
backend = provider.get_backend('ibmq_manila')
simulator = Aer.get_backend('qasm_simulator')
#In[4]:
#Number of input states
ntheta = 40
nphi = 40
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
        qrA = QuantumRegister(2)
        qrB = QuantumRegister(1)
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
        #mycircuit.h(qrA[1])
        #mycircuit.cx(qrA[1], qrB[0])
        #mycircuit.barrier()
            #Bell Measurement on alice1 and alice2 qubits
        mycircuit.cx(qrA[0], qrA[1])
        mycircuit.h(qrA[0])
        mycircuit.barrier()
            #Measurement on alice's qubits
        mycircuit.measure([qrA[0],qrA[1]], [0,1])
        mycircuit.barrier()
            #Deferred measurement - control operations on bob's qubit
        mycircuit.cx(qrA[1], qrB[0])
        mycircuit.cz(qrA[0], qrB[0])
        mycircuit.barrier()
            #Quantum State Tomography Circuits
        qstom = state_tomography_circuits(mycircuit, qrB[0])
        nshots = 1_000
    #REAL Device Execution
        qstcir = transpile(qstom, backend = backend, optimization_level=3, layout_method='sabre', routing_method='sabre')
        result = execute(qstcir, backend = backend, shots = nshots).result()
            #Data Fit and State fidelity computation
        qst_cir = StateTomographyFitter(result, qstom)
        sv_qst = qst_cir.fit(method='lstsq')
        sf = state_fidelity(sv, sv_qst, validate=False)
        
    #QASM Simulator Execution
        #result = execute(qstom, backend = simulator, optimization_level = 3,shots = nshots).result()
            #Data Fit and State fidelity computation
        #qst_cir = StateTomographyFitter(result, qstom)
        #sv_qst = qst_cir.fit(method='cvx')
        #sf = state_fidelity(sv, sv_qst, validate=False)
        #Fidelities sum
        avef += sf*sin(theta)
#Average fidelity
avet = avef/(4*pi)*dtheta*dphi
print(avet)