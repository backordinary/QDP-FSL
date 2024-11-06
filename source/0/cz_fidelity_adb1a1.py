# https://github.com/SotaIno/CircuitQEDsim/blob/564b7462e534f74e5a5a35ee4bd9989eb0efad37/CZ_fidelity.py
import os
import math
import numpy as np
import qutip as qt
import scipy
from scipy import constants
from scipy import interpolate
from scipy import integrate
import sympy as sym
from systemConst import ket,iniState2Qsys,Tunabletransmon,QQ
from CZwave import CZpulse

import matplotlib.pyplot as plt
import sys
import quantum_okiba as qo
from tqdm import tqdm
from qiskit import Aer
from qiskit.quantum_info import average_gate_fidelity
import fastadiabatic_net_zero as fn

pi = np.pi
e = constants.e # [C]
h = constants.h # [m^2 kg/s]
hbar = constants.hbar
iDir = os.path.abspath(os.path.dirname(__file__))
opts = qt.solver.Options(nsteps=10000)

def Hq(Nq, qFreq, qAnhar):
    Hqs = 0
    eigenFreq_list = [0,qFreq,2*qFreq-qAnhar]
    for i in range(Nq):
        Hqs = Hqs + eigenFreq_list[i] * ( ket(Nq, i) * ket(Nq, i).dag() )
    return Hqs

def MW_shaped(t,args):
    
    amp = args['mwamp']
    shape = args['shape'] 
    if int(t)>=len(shape):
        n=len(shape)-1
    else:
        n=int(t)
    return amp * shape[n]

def PhaseChange(state_list):
    
    final01 = [0] * len(state_list)
    final02 = [0] * len(state_list)
    final10 = [0] * len(state_list)
    final11 = [0] * len(state_list)

    pop11 = [0] * len(state_list)
    pop02 = [0] * len(state_list)

    phase10 = [0] * len(state_list)
    phase01 = [0] * len(state_list)
    phase11 = [0] * len(state_list)

    phaseDiff = [0] * len(state_list)
    
    for i in range(len(state_list)):
            
        final01[i] = state_list[i][:][1]
        final02[i] = state_list[i][:][2]
        final10[i] = state_list[i][:][3]
        final11[i] = state_list[i][:][4]

        pop11[i] = np.absolute(final11[i])**2
        pop02[i] = np.absolute(final02[i])**2
        
        phase01[i] = np.angle(final01[i]) / pi
        phase10[i] = np.angle(final10[i]) / pi
        phase11[i] = np.angle(final11[i]) / pi
        
        phaseDiff[i] = phase11[i] - phase10[i] - phase01[i]
        
        # phase ordering
        if i > 0 and phase10[i] - phase10[i-1] < -1:
            phase10[i] = phase10[i] + 2
        if i > 0 and phase10[i] - phase10[i-1] > 1:
            phase10[i] = phase10[i] - 2
            
        if i > 0 and phaseDiff[i] - phaseDiff[i-2] < -1:
            phaseDiff[i] = phaseDiff[i] + 2
        if i > 0 and phaseDiff[i] - phaseDiff[i-2] > 1:
            phaseDiff[i] = phaseDiff[i] - 2

    return phaseDiff,pop11,pop02

Ej1 = 17
Ej2 = 22
Ec1 = 0.27
Ec2 = 0.27
g=0.015 #/2pi

Q1=Tunabletransmon(EC=Ec1, EJmax=Ej1)
Q2=Tunabletransmon(EC=Ec2, EJmax=Ej2)
QQ=QQ(Q1,Q2,g)

#ini_state=iniState2Qsys(Q1.Nq,Q2.Nq,1,1)
Nq=3

ini_coeff = [0,1e-9,0,1e-9,1,0,0,0,0] # 11
ini_state = ini_coeff[0]*qt.tensor(ket(Nq,0), ket(Nq,0)) \
            + ini_coeff[1]*qt.tensor(ket(Nq,0), ket(Nq,1)) \
            + ini_coeff[2]*qt.tensor(ket(Nq,0), ket(Nq,2)) \
            + ini_coeff[3]*qt.tensor(ket(Nq,1), ket(Nq,0)) \
            + ini_coeff[4]*qt.tensor(ket(Nq,1), ket(Nq,1)) \
            + ini_coeff[5]*qt.tensor(ket(Nq,1), ket(Nq,2)) \
            + ini_coeff[6]*qt.tensor(ket(Nq,2), ket(Nq,0)) \
            + ini_coeff[7]*qt.tensor(ket(Nq,2), ket(Nq,1)) \
            + ini_coeff[8]*qt.tensor(ket(Nq,2), ket(Nq,2))

Iq1=qt.qeye(Nq)
Hq1_lab=QQ.Hq1*(2*pi)
rot2 = Hq(Q2.Nq, 0, abs(Q2.anh*(2*pi)))
q2Freqs = qt.qdiags(np.arange(0,Q2.Nq,1),0)
Hq2_t_ind = qt.tensor(Iq1, rot2) #Hq2_rot(constant term)
Hq2_t_dep = qt.tensor(Iq1, q2Freqs) #Hq2_rot(modulation term)
Hint = QQ.Hint12*(2*pi)
H_rot = [Hq1_lab + Hq2_t_ind + Hint, [Hq2_t_dep, MW_shaped]]
tgstart=48
tgend=48

for gt in range(tgstart,tgend+1):

    pulsesystem=CZpulse(Q1,Q2,g,the_f=0.88,lambda2=0.13,gatetime=gt)
    tg=pulsesystem.tg
    t_list,adiabaticpulse = pulsesystem.netzeropulse()
    degeneracy_freq=pulsesystem.qFreq20

    args = {'mwamp':1.0,'shape':adiabaticpulse}
    #res = qt.sesolve(H_rot, ini_state, t_list, e_ops=[], args=args, options=opts, progress_bar=None)
    _res = qt.propagator(H_rot, t_list, e_ops=[], args=args, options=opts, progress_bar=None)
    res = []
    for i in range(len(_res)):
        sp = qt.to_super(_res[i])
        res.append(sp)
    res = np.array(res)

    print(len(res))
    print(res[1].dims)
    print(res[1].shape)