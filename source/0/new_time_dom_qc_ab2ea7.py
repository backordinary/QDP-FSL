# https://github.com/JohnstonResearchGroup/Keen_etal_CCGF_2021/blob/97f7423e7e4384ceecd84a1b378a191af8802d38/Two_Site/new_time_dom_QC.py
import os
import sys
import math, cmath
import re

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
#from numpy import linalg as la
#from numpy.linalg import matrix_power
from scipy.linalg import cosm, expm, sinm
from qiskit import *
from qiskit.providers.aer import *
#from qiskit.circuit import Parameter
from qiskit.compiler import transpile
#from qiskit.extensions import initializer
#from qiskit.providers.aer.noise import NoiseMo#del
from qiskit.tools.visualization import circuit_drawer
#from qiskit.providers.aer.noise.errors import depolarizing_error
#from qiskit.quantum_info import (Pauli, basis_state, process_fi#delity,
#                                state_fi#delity)
#from qiskit.tools.visualization import circ_drawer, plot_histogram
#from scipy.misc import derivative
from scipy.optimize import (curve_fit, minimize)
from Cluster_ops import *
from Measurement_Circuits import *
from Pauli_matrices import *
from Params import *
from indexed import IndexedOrderedDict
import csv
from time import perf_counter
import warnings
warnings.filterwarnings("ignore", message='Timestamps in IBMQ backend properties, jobs, and job results are all now in local time instead of UTC.')
V = float(sys.argv[1])
'''
if (second_order):
    print('Data for U=',U,' and V=',V, '. With second order product formula.', '\n')
else:
    print('Data for U=',U,' and V=',V, '. With first order product formula.', '\n')
print('Shots = ', shot,'.', 'Shot error = ', 1.0/np.sqrt(shot), '\n')
'''
print('Shots = ', shot,'.', 'Shot error = ', 1.0/np.sqrt(shot), '\n')
t=t21_val(U,V)
print('t21 = ', t)
print('Error probabilities: ', '\n', 'Single qubit: ', prob_1, '\n', 'CNOT: ',\
prob_2)
#sigma_sing_ex = lam12_val(U,V)*(c2dag@c1 - c1dag@c2 + c3dag@c4 - c4dag@c3)
#sigma_doub_ex = (lamdoub_val(U,V) - lam12_val(U,V)**2).real*(c2dag@c3dag@c4@c1 - c1dag@c4dag@c3@c2)


H_SIAM =  U/4.0 * (Z1Z3 - Z1 - Z3 + I) + mu/2.0 * (Z1 + Z3) \
+V/2.0 * (X1X2 + Y1Y2 + X3X4 + Y3Y4) - U/2.0 * I


def E0N(hamiltonian, trial_vec, u, v):
    return (np.conjugate(trial_vec).transpose()@expm(-\
    T_op(u,v))@hamiltonian@expm(T_op(u,v))@trial_vec)

def parity_of_ancilla(bitstring, ancilla_index):
    if (bitstring[-(ancilla_index+1)] == '0'):
        return 1
    elif (bitstring[-(ancilla_index+1)]=='1'):
        return -1
    else:
        raise ValueError('Problem with the input bits')


#IBMQ.#delete_account()
#IBMQ.save_account('15aa254eef157eebed177319173619e986b3927b87f7df5243b3f58da1a1c8507a220ef1ef1ac670e6c803641b28934d7a13720c2ee6488858bfda8b58a3ce46')
#provider = IBMQ.load_account()
ornl_provider = IBMQ.load_account()
#device = ornl_provider.get_backend('ibmq_johannesburg')
#properties = device.properties()
#coupling_map = device.configuration().coupling_map
#backend = ornl_provider.get_backend('ibmq_qasm_simulator')
backend = Aer.get_backend('qasm_simulator')



Ecc = E0N(H_SIAM, trial_state, U, V)
print('ECC: ', Ecc)

all_term_Gr_impt=np.zeros(steps+1,dtype=complex)
Gr_impt_3term=np.zeros(steps+1,dtype=complex)

X3UrX3=np.zeros(steps+1,dtype=complex)
X3UrX1X2X3=np.zeros(steps+1,dtype=complex)
X1X2X3UrX3=np.zeros(steps+1,dtype=complex)
X1X2X3UrX1X2X3=np.zeros(steps+1,dtype=complex)

X3UaX3=np.zeros(steps+1,dtype=complex)
X3UaX1X2X3=np.zeros(steps+1,dtype=complex)
X1X2X3UaX3=np.zeros(steps+1,dtype=complex)
X1X2X3UaX1X2X3=np.zeros(steps+1,dtype=complex)

Ur_expecval=np.zeros(steps+1, dtype=complex)

eiecc_val=np.zeros(steps+1, dtype=complex)

system_reg = QuantumRegister(4)
ancilla_reg1 = QuantumRegister(1)
measurement_reg = ClassicalRegister(5)
for nt in range(0, steps+1):

    print('Time step ', nt, ' of ', steps)

    Gr_real_circs = []
    Ga_real_circs = []
    Ua_term=False

    time_evol_circ = QuantumCircuit(ancilla_reg1, system_reg, measurement_reg)
    time_evol_circ.h(ancilla_reg1[0])
    #time_evol_circ.s(ancilla_reg1[0])
    trial_state_prep(time_evol_circ, system_reg[0], system_reg[1], system_reg[2],system_reg[3])
    full_trotter_step(nt, num_trot, Ua_term, U, V, Ecc, time_evol_circ, ancilla_reg1[0], system_reg[0], system_reg[1], system_reg[2], system_reg[3])

    time_evol_circ.h(ancilla_reg1[0])

    time_evol_circ.measure(ancilla_reg1[0], measurement_reg[0])
    time_evol_circ.measure(system_reg[0], measurement_reg[4])
    time_evol_circ.measure(system_reg[1], measurement_reg[3])
    time_evol_circ.measure(system_reg[2], measurement_reg[2])
    time_evol_circ.measure(system_reg[3], measurement_reg[1])

    time_evol_job = execute(time_evol_circ,\
    backend,basis_gates=basis_gates, optimization_level=1, shots=shot)
    #print("Job ID: " + str(GrA33_real_job.job_id()))

    time_evol_result = time_evol_job.result()

    #eiecc=np.exp(-twopi*1j*(-Ecc - U/4.0)*dt*nt)
    #aeiecc=np.exp(-twopi*1j*(Ecc+U/4.0)*dt*nt)
    #eiecc_val[nt]=eiecc

    time_evol_counts = time_evol_result.get_counts(time_evol_circ)

    time_evol_keys = time_evol_counts.keys()
    #par_anc=0
    time_evol_expecval=0.0
    for key in time_evol_keys:
        par_anc = parity_of_ancilla( key, 0 )
        #print("Parity of ancilla: ", par_anc)
        #print('h2')
        #if abs(par_anc+1.0) < 1.0e-2:
        #    eiecc=1.0
            #print('h1')
        #else:
        #    eiecc=1.0
        time_evol_expecval += (time_evol_counts[key]/shot) * \
        (par_anc)


    Ur_expecval[nt] += time_evol_expecval

    Gr_X3UrX3 = QuantumCircuit(ancilla_reg1, system_reg, measurement_reg)

    Gr_X3UrX3.h(ancilla_reg1[0])
    #GrA_X3UrX3.s(ancilla_reg1[0])

    trial_state_prep(Gr_X3UrX3, system_reg[0], system_reg[1], system_reg[2],\
    system_reg[3])
    #Gr_X3UrX3.cz(ancilla_reg1[0],system_reg[0])
    #Gr_X3UrX3.cz(ancilla_reg1[0],system_reg[1])
    Gr_X3UrX3.x(system_reg[2])

    full_trotter_step(nt, num_trot, Ua_term, U, V, Ecc, Gr_X3UrX3, ancilla_reg1[0], \
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])

    #Gr_X3UrX3.cz(ancilla_reg1[0],system_reg[0])
    #Gr_X3UrX3.cz(ancilla_reg1[0],system_reg[1])
    #Gr_X3UrX3.cx(ancilla_reg1[0],system_reg[2])

    Gr_X3UrX3.sdg(ancilla_reg1[0])
    Gr_X3UrX3.h(ancilla_reg1[0])

    Gr_X3UrX3.measure(ancilla_reg1[0], measurement_reg[0])
    Gr_X3UrX3.measure(system_reg[0], measurement_reg[4])
    Gr_X3UrX3.measure(system_reg[1], measurement_reg[3])
    Gr_X3UrX3.measure(system_reg[2], measurement_reg[2])
    Gr_X3UrX3.measure(system_reg[3], measurement_reg[1])

    if(nt==1):
        print('Operation count for < X3 Ur X3 > at time step 1: ', transpile(Gr_X3UrX3, backend=backend, basis_gates=basis_gates,  optimization_level=2).count_ops())
        #fig=plt.figure()
        #circuit_drawer(Gr_X3UrX3, output='mpl')
        #plt.savefig('expecvalX3UrX3_circ.pdf')
        #plt.close()
    else:
        pass

    Gr_real_circs.append(Gr_X3UrX3)

    Gr_X3UrX1X2X3 = QuantumCircuit(ancilla_reg1, system_reg, measurement_reg)

    Gr_X3UrX1X2X3.h(ancilla_reg1[0])
    #Gr_X3UrX1X2X3.s(ancilla_reg1[0])
    trial_state_prep(Gr_X3UrX1X2X3, system_reg[0], system_reg[1], \
    system_reg[2], system_reg[3])
    Gr_X3UrX1X2X3.x(system_reg[2])
    #Gr_X3UrX1X2X3.cz(ancilla_reg1[0],system_reg[0])
    #Gr_X3UrX1X2X3.cz(ancilla_reg1[0],system_reg[1])

    full_trotter_step(nt, num_trot, Ua_term, U, V,Ecc, Gr_X3UrX1X2X3, ancilla_reg1[0], \
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])
    Gr_X3UrX1X2X3.cx(ancilla_reg1[0],system_reg[0])
    Gr_X3UrX1X2X3.cx(ancilla_reg1[0],system_reg[1])
    #Gr_X3UrX1X2X3.x(system_reg[2])
    Gr_X3UrX1X2X3.sdg(ancilla_reg1[0])
    Gr_X3UrX1X2X3.h(ancilla_reg1[0])


    Gr_X3UrX1X2X3.measure(ancilla_reg1[0], measurement_reg[0])
    Gr_X3UrX1X2X3.measure(system_reg[0], measurement_reg[4])
    Gr_X3UrX1X2X3.measure(system_reg[1], measurement_reg[3])
    Gr_X3UrX1X2X3.measure(system_reg[2], measurement_reg[2])
    Gr_X3UrX1X2X3.measure(system_reg[3], measurement_reg[1])

    Gr_real_circs.append(Gr_X3UrX1X2X3)

    Gr_X1X2X3UrX3 = QuantumCircuit(ancilla_reg1, system_reg, measurement_reg)

    Gr_X1X2X3UrX3.h(ancilla_reg1[0])
    #Gr_X1X2X3UrX3.s(ancilla_reg1[0])
    trial_state_prep(Gr_X1X2X3UrX3, system_reg[0], system_reg[1], \
    system_reg[2], system_reg[3])
    Gr_X1X2X3UrX3.x(system_reg[2])
    Gr_X1X2X3UrX3.cx(ancilla_reg1[0],system_reg[0])
    Gr_X1X2X3UrX3.cx(ancilla_reg1[0],system_reg[1])


    full_trotter_step(nt, num_trot, Ua_term, U, V,Ecc, Gr_X1X2X3UrX3, ancilla_reg1[0], \
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])
    Gr_X1X2X3UrX3.cz(ancilla_reg1[0],system_reg[0])
    Gr_X1X2X3UrX3.cz(ancilla_reg1[0],system_reg[1])
    #Gr_X1X2X3UrX3.x(ancilla_reg1[0], system_reg[2])

    Gr_X1X2X3UrX3.sdg(ancilla_reg1[0])
    Gr_X1X2X3UrX3.h(ancilla_reg1[0])


    Gr_X1X2X3UrX3.measure(ancilla_reg1[0], measurement_reg[0])
    Gr_X1X2X3UrX3.measure(system_reg[0], measurement_reg[4])
    Gr_X1X2X3UrX3.measure(system_reg[1], measurement_reg[3])
    Gr_X1X2X3UrX3.measure(system_reg[2], measurement_reg[2])
    Gr_X1X2X3UrX3.measure(system_reg[3], measurement_reg[1])

    Gr_real_circs.append(Gr_X1X2X3UrX3)

    Gr_X1X2X3UrX1X2X3 = QuantumCircuit(ancilla_reg1, system_reg, \
    measurement_reg)

    Gr_X1X2X3UrX1X2X3.h(ancilla_reg1[0])
    #Gr_X1X2X3UrX1X2X3.s(ancilla_reg1[0])
    trial_state_prep(Gr_X1X2X3UrX1X2X3, system_reg[0], system_reg[1], \
    system_reg[2], system_reg[3])
    Gr_X1X2X3UrX1X2X3.x(system_reg[0])
    Gr_X1X2X3UrX1X2X3.x(system_reg[1])
    Gr_X1X2X3UrX1X2X3.x(system_reg[2])

    full_trotter_step(nt, num_trot, Ua_term, U, V,Ecc, Gr_X1X2X3UrX1X2X3, ancilla_reg1[0],\
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])

    Gr_X1X2X3UrX1X2X3.sdg(ancilla_reg1[0])
    Gr_X1X2X3UrX1X2X3.h(ancilla_reg1[0])

    Gr_X1X2X3UrX1X2X3.measure(ancilla_reg1[0], measurement_reg[0])
    Gr_X1X2X3UrX1X2X3.measure(system_reg[0], measurement_reg[4])
    Gr_X1X2X3UrX1X2X3.measure(system_reg[1], measurement_reg[3])
    Gr_X1X2X3UrX1X2X3.measure(system_reg[2], measurement_reg[2])
    Gr_X1X2X3UrX1X2X3.measure(system_reg[3], measurement_reg[1])

    Gr_real_circs.append(Gr_X1X2X3UrX1X2X3)


    Gr_real_job = execute(Gr_real_circs,backend=backend,basis_gates=basis_gates, optimization_level=1, shots=shot)
    #print("Job ID: " + str(GrA33_real_job.job_id()))

    Gr_real_result = Gr_real_job.result()

    #eiecc=np.exp(twopi*1j*(Ecc + U/4.0)*dt*nt)
    #aeiecc=np.exp(-twopi*1j*(Ecc + U/4.0)*dt*nt)

    for j in np.arange(0,4,1):
        Gr_real_counts = Gr_real_result.get_counts(Gr_real_circs[j])

        Gr_real_keys = Gr_real_counts.keys()

        Gr_real_expecval=0.0
        for key in Gr_real_keys:
            Gr_real_expecval += (Gr_real_counts[key]/shot) * \
            (parity_of_ancilla( key, 0 ))
        if j==0:
            X3UrX3[nt]=Gr_real_expecval
            #print('help1')
        elif j==1:
            X3UrX1X2X3[nt]=Gr_real_expecval
            #print('help2')
        elif j==2:
            X1X2X3UrX3[nt]=Gr_real_expecval
            #print('help3')
        elif j==3:
            X1X2X3UrX1X2X3[nt]=Gr_real_expecval
            #print('help4')
        else:
            raise ValueError('Problem calculating expecvals')
    '''
    Ua_term=False

    Ga_X3UaX3 = QuantumCircuit(ancilla_reg1, system_reg, measurement_reg)

    Ga_X3UaX3.h(ancilla_reg1[0])
    Ga_X3UaX3.s(ancilla_reg1[0])
    trial_state_prep(Ga_X3UaX3, system_reg[0], system_reg[1], system_reg[2],\
    system_reg[3])
    Ga_X3UaX3.cz(ancilla_reg1[0], system_reg[0])
    Ga_X3UaX3.cz(ancilla_reg1[0], system_reg[1])
    Ga_X3UaX3.cx(ancilla_reg1[0], system_reg[2])

    full_trotter_step(nt, Ua_term, U, V, Ga_X3UaX3, ancilla_reg1[0], \
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])
    Ga_X3UaX3.cz(ancilla_reg1[0], system_reg[0])
    Ga_X3UaX3.cz(ancilla_reg1[0], system_reg[1])
    Ga_X3UaX3.cx(ancilla_reg1[0], system_reg[2])

    #Ga_X3UaX3.sdg(ancilla_reg1[0])
    Ga_X3UaX3.h(ancilla_reg1[0])

    Ga_X3UaX3.measure(ancilla_reg1[0], measurement_reg[0])
    Ga_X3UaX3.measure(system_reg[0], measurement_reg[4])
    Ga_X3UaX3.measure(system_reg[1], measurement_reg[3])
    Ga_X3UaX3.measure(system_reg[2], measurement_reg[2])
    Ga_X3UaX3.measure(system_reg[3], measurement_reg[1])

    Ga_real_circs.append(Ga_X3UaX3)

    Ga_X3UaX1X2X3 = QuantumCircuit(ancilla_reg1, system_reg, measurement_reg)

    Ga_X3UaX1X2X3.h(ancilla_reg1[0])
    Ga_X3UaX1X2X3.s(ancilla_reg1[0])
    trial_state_prep(Ga_X3UaX1X2X3, system_reg[0], system_reg[1], \
    system_reg[2], system_reg[3])
    Ga_X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[0])
    Ga_X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[1])
    Ga_X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[2])

    full_trotter_step(nt, Ua_term, U, V, Ga_X3UaX1X2X3, ancilla_reg1[0], \
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])
    Ga_X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[2])
    Ga_X3UaX1X2X3.cz(ancilla_reg1[0], system_reg[0])
    Ga_X3UaX1X2X3.cz(ancilla_reg1[0], system_reg[1])

    #Ga_X3UaX1X2X3.sdg(ancilla_reg1[0])
    Ga_X3UaX1X2X3.h(ancilla_reg1[0])

    Ga_X3UaX1X2X3.measure(ancilla_reg1[0], measurement_reg[0])
    Ga_X3UaX1X2X3.measure(system_reg[0], measurement_reg[4])
    Ga_X3UaX1X2X3.measure(system_reg[1], measurement_reg[3])
    Ga_X3UaX1X2X3.measure(system_reg[2], measurement_reg[2])
    Ga_X3UaX1X2X3.measure(system_reg[3], measurement_reg[1])

    Ga_real_circs.append(Ga_X3UaX1X2X3)

    Ga_X1X2X3UaX3 = QuantumCircuit(ancilla_reg1, system_reg, measurement_reg)

    Ga_X1X2X3UaX3.h(ancilla_reg1[0])
    Ga_X1X2X3UaX3.s(ancilla_reg1[0])
    trial_state_prep(Ga_X1X2X3UaX3, system_reg[0], system_reg[1], \
    system_reg[2], system_reg[3])
    Ga_X1X2X3UaX3.cx(ancilla_reg1[0], system_reg[2])
    Ga_X1X2X3UaX3.cz(ancilla_reg1[0], system_reg[0])
    Ga_X1X2X3UaX3.cz(ancilla_reg1[0], system_reg[1])

    full_trotter_step(nt, Ua_term, U, V, Ga_X1X2X3UaX3, ancilla_reg1[0], \
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])
    Ga_X1X2X3UaX3.cx(ancilla_reg1[0], system_reg[0])
    Ga_X1X2X3UaX3.cx(ancilla_reg1[0], system_reg[1])
    Ga_X1X2X3UaX3.cx(ancilla_reg1[0], system_reg[2])

    #Ga_X1X2X3UaX3.sdg(ancilla_reg1[0])
    Ga_X1X2X3UaX3.h(ancilla_reg1[0])

    Ga_X1X2X3UaX3.measure(ancilla_reg1[0], measurement_reg[0])
    Ga_X1X2X3UaX3.measure(system_reg[0], measurement_reg[4])
    Ga_X1X2X3UaX3.measure(system_reg[1], measurement_reg[3])
    Ga_X1X2X3UaX3.measure(system_reg[2], measurement_reg[2])
    Ga_X1X2X3UaX3.measure(system_reg[3], measurement_reg[1])

    Ga_real_circs.append(Ga_X1X2X3UaX3)

    Ga_X1X2X3UaX1X2X3 = QuantumCircuit(ancilla_reg1, system_reg, \
    measurement_reg)

    Ga_X1X2X3UaX1X2X3.h(ancilla_reg1[0])
    Ga_X1X2X3UaX1X2X3.s(ancilla_reg1[0])
    trial_state_prep(Ga_X1X2X3UaX1X2X3, system_reg[0], system_reg[1], \
    system_reg[2], system_reg[3])
    Ga_X1X2X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[0])
    Ga_X1X2X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[1])
    Ga_X1X2X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[2])

    full_trotter_step(nt, Ua_term, U, V, Ga_X1X2X3UaX1X2X3, ancilla_reg1[0], \
    system_reg[0], system_reg[1], system_reg[2], system_reg[3])
    Ga_X1X2X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[0])
    Ga_X1X2X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[1])
    Ga_X1X2X3UaX1X2X3.cx(ancilla_reg1[0], system_reg[2])

    #Ga_X1X2X3UaX1X2X3.sdg(ancilla_reg1[0])
    Ga_X1X2X3UaX1X2X3.h(ancilla_reg1[0])

    Ga_X1X2X3UaX1X2X3.measure(ancilla_reg1[0], measurement_reg[0])
    Ga_X1X2X3UaX1X2X3.measure(system_reg[0], measurement_reg[4])
    Ga_X1X2X3UaX1X2X3.measure(system_reg[1], measurement_reg[3])
    Ga_X1X2X3UaX1X2X3.measure(system_reg[2], measurement_reg[2])
    Ga_X1X2X3UaX1X2X3.measure(system_reg[3], measurement_reg[1])

    Ga_real_circs.append(Ga_X1X2X3UaX1X2X3)


    Ga_real_job = execute(Ga_real_circs,backend,basis_gates=basis_gates, optimization_level=1, shots=shot)
    #print("Job ID: " + str(GrA33_real_job.job_id()))

    Ga_real_result = Ga_real_job.result()

    for j in np.arange(0,4,1):
        Ga_real_counts = Ga_real_result.get_counts(Ga_real_circs[j])

        Ga_real_keys = Ga_real_counts.keys()

        Ga_real_expecval=0.0
        for key in Ga_real_keys:
            Ga_real_expecval += (Ga_real_counts[key]/shot)*(parity_of_ancilla( key, 0 ))
        if j==0:
            X3UaX3[nt]=aeiecc*Ga_real_expecval
        elif j==1:
            X3UaX1X2X3[nt]=aeiecc*Ga_real_expecval
        elif j==2:
            X1X2X3UaX3[nt]=aeiecc*Ga_real_expecval
        elif j==3:
            X1X2X3UaX1X2X3[nt]=aeiecc*Ga_real_expecval
        else:
            raise ValueError('Problem calculating expecvals')
    '''
print('Compiling Results...', '\n')

all_Ur_only = np.zeros(steps+1, dtype=complex)
Ur_only_3term = np.zeros(steps+1, dtype=complex)
for i in np.arange(0, steps+1):
    #all_term_Gr_impt[i] = complex(0.0,nu1*mu1*X3UrX3[i] + nu1*mu2*X3UrX1X2X3[i]\
#             + nu2*mu2*X1X2X3UrX1X2X3[i]\
#            +nu2*mu1*X1X2X3UrX3[i]\
#            -nu1*mu1*X3UaX3[i]\
#                -nu2*mu1*X1X2X3UaX3[i] - nu2*mu2*X1X2X3UaX1X2X3[i]\
#                - nu1*mu2*X3UaX1X2X3[i])
#    Gr_impt_3term[i] = complex(0.0,nu1*mu1*X3UrX3[i] + nu1*mu2*X3UrX1X2X3[i]\
#                 + nu2*mu2*X1X2X3UrX1X2X3[i]\
#                -nu1*mu1*X3UaX3[i]\
#                -nu1*mu2*X3UaX1X2X3[i] - nu2*mu2*X1X2X3UaX1X2X3[i])
    all_Ur_only[i] = 1j*(nu1*mu1*X3UrX3[i] + nu1*mu2*X3UrX1X2X3[i]\
                 + nu2*mu2*X1X2X3UrX1X2X3[i]\
                 +nu2*mu1*X1X2X3UrX3[i])
    Ur_only_3term[i] = 1j*(nu1*mu1*X3UrX3[i] + nu1*mu2*X3UrX1X2X3[i]\
                     + nu2*mu2*X1X2X3UrX1X2X3[i])

#popt, pcov = curve_fit(fit2, time_arr, eiecc_val.imag, p0=8, maxfev=100000)
#print('popt: ', popt)
print('Re<X3UrX3>: ', X3UrX3.real, '\n')
print('Re<X3UrX1X2X3>: ', X3UrX1X2X3.real, '\n')
print('Re<X1X2X3UrX3>: ', X1X2X3UrX3.real, '\n')
print('Re<X1X2X3UrX1X2X3>: ', X1X2X3UrX1X2X3.real, '\n')
'''
print('Im<X3UrX3>: ', X3UrX3.imag, '\n')
print('Im<X3UrX1X2X3>: ', X3UrX1X2X3.imag, '\n')
print('Im<X1X2X3UrX3>: ', X1X2X3UrX3.imag, '\n')
print('Im<X1X2X3UrX1X2X3>: ', X1X2X3UrX1X2X3.imag, '\n')
'''
print('<Ur>: ', Ur_expecval.real)
#print('<X3UaX3>: ', X3UaX3, '\n')
#print('<X3UaX1X2X3>: ', X3UaX1X2X3, '\n')
#print('<X1X2X3UaX3>: ', X1X2X3UaX3, '\n')
#print('<X1X2X3UaX1X2X3>: ', X1X2X3UaX1X2X3, '\n')

output = np.column_stack(((time_arr.real).flatten(),\
#all_term_Gr_impt.flatten(),Gr_impt_3term.flatten(),
all_Ur_only.flatten(), Ur_only_3term.flatten(), X3UrX3.real.flatten(), X3UrX1X2X3.real.flatten(), X1X2X3UrX1X2X3.real.flatten()))

np.savetxt('Time_Domain_Data/Quantum/new_QC_G_impt_V=%.2f_%d_steps_tot_time3=%.1f_error_rates=(%.2e,%.2e).csv' % (V, steps,total_time, prob_1, prob_2),output, header="time, \
all_Ur_only, Ur_only_3term, <X3UrX3>, <X3UrX1X2X3>, <X1X2X3UrX1X2X3>", delimiter='\t')

#f=plt.figure(1)
#plt.plot(time_arr,Ur_expecval, label='<Ur>')
#plt.legend()
#plt.show()
'''
g=plt.figure(2)
plt.plot(time_arr, eiecc_val, 'o', markersize=4)
plt.plot(fine_time_arr, fit2(fine_time_arr, popt[0]))
plt.show()
'''
quit()
'''
f=plt.figure(1)
plt.plot(time_arr, all_term_Gr_impt.imag, label='All')
plt.plot(time_arr, Gr_impt_3term.imag, label='Three terms')
plt.legend()
g=plt.figure(2)
plt.plot(time_arr, all_Ur_only.imag, label='All Ur')
plt.plot(time_arr, Ur_only_3term.imag, label='Three terms Ur')
plt.legend()
plt.show()
'''
#plt.close()
