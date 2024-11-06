# https://github.com/SotaIno/CircuitQEDsim/blob/564b7462e534f74e5a5a35ee4bd9989eb0efad37/Idialcirc.py
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
#from CZwave import CZpulse

import matplotlib.pyplot as plt
import sys
import quantum_okiba as qo
from tqdm import tqdm
from qiskit import QuantumCircuit,visualization,Aer,execute
from qiskit.quantum_info import average_gate_fidelity

pi = np.pi
e = constants.e # [C]
h = constants.h # [m^2 kg/s]
hbar = constants.hbar
iDir = os.path.abspath(os.path.dirname(__file__))
opts = qt.solver.Options(nsteps=10000)

#回路を設計
circ = QuantumCircuit(2)
circ.h(0)
Image=circ.draw(output='mpl')
Image.savefig(iDir+'/sfg.png')

#回路を実行1
backend = Aer.get_backend('statevector_simulator')
job = execute(circ, backend) #これでstatusとresultを得られる状態に

result = job.result()
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

#回路を実行2
backend = Aer.get_backend('unitary_simulator')
job = execute(circ, backend) #これでstatusとresultを得られる状態に

result = job.result()
print(result.get_unitary(circ, decimals=3))