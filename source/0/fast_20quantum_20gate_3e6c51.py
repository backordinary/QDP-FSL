# https://github.com/LilyHeAsamiko/QC/blob/216a52fb15464b238ca8f3903748b745af8f7682/Fast%20Quantum%20Gate/fast%20quantum%20gate.py
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 06:17:12 2020

@author: LilyHeAsamiko
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:10:06 2020

@author: LilyHeAsamiko
"""
#The quantum phase estimation algorithm uses phase kickback to write the phase of #U(in the Fourier basis) to the t qubits in the counting register.
#We then use the inverse QFT to translate this from the Fourier basis(estimates  θ in U|ψ⟩=e2πiθ|ψ⟩ . Here |ψ⟩  is an eigenvector and  e2πiθ  is the corresponding eigenvalue.) into the computational basis, which we can measure.
# propose a fast phase gate for neutral trapped atoms, corresponding to a truth table
#|ǫ 1 i ⊗ |ǫ 2 i → e iǫ 1 ǫ 2 ϕ |ǫ 1 i ⊗ |ǫ 2 i for the logical states |ǫ i i
#with ǫ i = 0,1, 
#which (i) exploits the very large interac-
#tions of permanent dipole moments of laser excited Ryd-
#berg states in a constant electric field to entangle atoms,
#while (ii) allowing gate operation times set by the time
#scale of the laser excitation or the two particle interac-
#tion energy, which can be significantly shorter than the
#trap period

#!Note that: For electric fields below the Ingris-Teller limit
#the mixing of adjacent n-manifolds can be neglected,
#and the energy levels are split according to ∆E nqm =
#3nqea_0*E/2 with parabolic and magnetic quantum num-
#bers q = n − 1 − |m|,n − 3 − |m|,...,−(n − 1 − |m|)
#and m, respectively, e the electron charge, and a 0 the
#Bohr radius. These Stark states have permanent dipole
#moments µ ≡ µ_z*e_z = 3nqea_0*e_z/2
#(Rydberg states of a hydrogen atom within a given
#manifold of a fixed principal quantum number n are de-
#generate. This degeneracy is removed by applying a
#constant electric field E along the z-axis (linear Stark
#effect). 

#%matplotlib inline

#%matplotlib auto

import qiskit
#qiskit.__version__
from qiskit import IBMQ
IBMQ.load_account()
#MY_API_TOKEN = 'a93830f80226030329fc4e2e4d78c06bdf1942ce349fcf8f5c8021cfe8bd5abb01e4205fbd7b9c34f0b26bd335de7f1bcb9a9187a2238388106d16c6672abea2'
#provider = IBMQ.enable_account(MY_API_TOKEN)

from qiskit.compiler import transpile, assemble,schedule
from qiskit.tools.jupyter import *
from qiskit.visualization import *

import matplotlib.pyplot as plt

import numpy as np
import math
import os
import sys
import io
import requests
import urllib
#store pi amplitudes
#given the drive and target indices, and the option to either start with the drive qubit in the ground or excited state, returns a list of experiments for observing the oscillations.
from IPython import display
import time
import pandas as pd

# importing Qiskit
import qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, execute
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate

# import basic plot tools
from qiskit.visualization import plot_histogram

from random import *
from qiskit.visualization.bloch import Bloch
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import qiskit.pulse as pulse
import qiskit.pulse.pulse_lib as pulse_lib
from qiskit.pulse.pulse_lib import Gaussian, GaussianSquare
from qiskit.compiler import assemble
from qiskit.ignis.characterization.calibrations import rabi_schedules, RabiFitter
#from qiskit.pulse.commands import SamplePulse
from qiskit.pulse import *
from qiskit.tools.monitor import job_monitor
# function for constructing duffing models
from qiskit.providers.aer.pulse import duffing_system_model

#We will experimentally find a π-pulse for each qubit using the following procedure: 
#- A fixed pulse shape is set - in this case it will be a Gaussian pulse. 
#- A sequence of experiments is run, each consisting of a Gaussian pulse on the qubit, followed by a measurement, with each experiment in the sequence having a subsequently larger amplitude for the Gaussian pulse. 
#- The measurement data is fit, and the pulse amplitude that completely flips the qubit is found (i.e. the π-pulse amplitude).
import warnings
warnings.filterwarnings('ignore')
from qiskit.tools.jupyter import *
get_ipython().run_line_magic('matplotlib', 'inline')

provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"
#from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import PulseSimulator
from scipy.optimize import curve_fit
import math

# import basic plot tools
from qiskit.visualization import plot_histogram


def classify(point: complex):
    """Classify the given state as |0> or |1>."""
    def distance(a, b):
        return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)
    return int(distance(point, mean_exc) < distance(point, mean_gnd))

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    # Initialise all qubits to |+>
    for q in qubits:
        qc.h(q)
    return qc

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s

def Uw():
    Uw = np.aray([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    return Uw

def qft(n):
    """Creates an n-qubit QFT circuit"""
    circuit = QuantumCircuit(4)
    def swap_registers(circuit, n):
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        return circuit
    def qft_rotations(circuit, n):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cu1(np.pi/2**(n-qubit), qubit, n)
        qft_rotations(circuit, n)
    
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def get_job_data(job, average):
    """Retrieve data from a job that has already run.
    Args:
        job (Job): The job whose data you want.
        average (bool): If True, gets the data assuming data is an average.
                        If False, gets the data assuming it is for single shots.
    Return:
        list: List containing job result data. 
    """
    job_results = job.result(timeout=120) # timeout parameter set to 120 s
    result_data = []
    for i in range(len(job_results.results)):
        if average: # get avg data
            result_data.append(job_results.get_memory(i)[qubit]*scale_factor) 
        else: # get single data
            result_data.append(job_results.get_memory(i)[:, qubit]*scale_factor)  
    return result_data

def get_closest_multiple_of_4(num):
    """Compute the nearest multiple of 4. Needed because pulse enabled devices require 
    durations which are multiples of 4 samples.
    """
    return (int(num) - (int(num)%4))

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(n//2):
        circ.swap(qubit, n-qubit-1)
    for j in range(n):
        circ.cx(j, j+1)
        circ.h(j)

# center data around 0
def baseline_remove(values):
    return np.array(values) - np.mean(values)

def motion(t, rep):
    # assume t = 2， rep = 1
    # circuit estimates the phase of a unitary operator U. It estimates θ in ψ⟩=e2πiθ|ψ⟩ , where  |ψ⟩  is an eigenvector and  e2πiθ  is the corresponding eigenvalue. The circuit operates in the following steps
    # qubits 0 to 1 as counting qubits, and qubit 2 as the eigenstate of the unitary operator (T)
    
    #i. Setup:  |ψ⟩  is in one set of qubit registers. (Set it as 1)
    qpe = QuantumCircuit(t+1, t) #(3, 2)
    qpe.x(t)
    
    #ii. Superposition: Apply a n-bit Hadamard gate operation  
    # H⊗n on the counting register:
    for qubit in range(t):
        qpe.h(qubit)
    #Controlled Unitary Operations: We need to introduce the controlled X gate 
    #C−X that applies the unitary operator U on the target register only if its corresponding control bit is  
    #|1⟩ . Since U is a unitary operatory with eigenvector |ψ⟩  such that U|ψ⟩=eπiθ|ψ⟩ , this means:
    # repetitions = 1
    for counting_qubit in range(t):
        for i in range(rep):
            #qpe.cx(math.pi/4, counting_qubit, 3); This is C-U
            qpe.cx(counting_qubit, counting_qubit +1); 
        rep *= 2     
    # measure the counting register    
    qpe.barrier()
    qpe.draw()
    return qpe

p0 = 2.5*10**(-7)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
kHz = 1.0e3 # Khertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
gamma = 100*kHz
u = 1.8*GHz
O0 = 100*MHz
delta0 = 1.7*GHz

qpe0 = motion(2, 1)
#qpe.draw() 
qpe0.barrier()
qpe0.measure(0,0)
qpe0.measure(1,1)

backend = Aer.get_backend('qasm_simulator')
shots = 512
results0 = execute(qpe0, backend=backend, shots=shots).result()

# perform the controlled unitary operations
answer0 = results0.get_counts()
n00 = answer0['00']
n01 = answer0['01']
n10 = answer0['10']
n11 = answer0['11']
p_0 =(2*n01+1)*p0 #qubit0 excitation probability
p_1 =(2*n11+1)*p0 #qubit1 excitation probability
g0 = (2*n00+1)*p0
g1 = (2*n10+1)*p0
e0 = p_0
e1 = p_1

qpe = motion(2, 1)
#qpe.draw() 
qpe.barrier() 
# Apply inverse QFT
qft_dagger(qpe, 2)
#qpe.draw()  
# Measure
qpe.barrier()
qpe.measure(0,0)
qpe.measure(1,1)
qpe.draw()   

#     ┌───┐                ░         ┌───┐     ┌─┐   
#q_0: ┤ H ├──■─────────────░──X───■──┤ H ├─────┤M├───
#     ├───┤┌─┴─┐           ░  │ ┌─┴─┐└───┘┌───┐└╥┘┌─┐
#q_1: ┤ H ├┤ X ├──■────■───░──X─┤ X ├──■──┤ H ├─╫─┤M├
#     ├───┤└───┘┌─┴─┐┌─┴─┐ ░    └───┘┌─┴─┐└───┘ ║ └╥┘
#q_2: ┤ X ├─────┤ X ├┤ X ├─░─────────┤ X ├──────╫──╫─
#     └───┘     └───┘└───┘ ░         └───┘      ║  ║ 
#c: 2/══════════════════════════════════════════╩══╩═
#                                                       0  1 

backend = Aer.get_backend('qasm_simulator')
shots = 512
results = execute(qpe, backend=backend, shots=shots).result()

# perform the controlled unitary operations
answer = results.get_counts()

plot_histogram(answer)
nt0 = answer['00']
nt1 = answer['10']
pt = nt1/shots
pt_0 = (2*nt0**2+2*(n00+n01)+1)*pt/shots**2
pt_1 = (2*nt1**2+2*(n10+n11)+1)*pt/shots**2

d0 = p_0
d1 = p_1
V0 = d0*delta0
V1 = d1*delta0

#Model A:
#Consider two atoms 1 and 2 at fixed positions (see and initially prepared in Stark eigenstates, with
#a dipole moment along z and a given m, as selected by the polarization of the laser exciting the Rydberg states from the ground state.)
hc=0
H_0 = u+(d0-gamma)-O0/2*(g0+hc)
H_1 = u+(d1-gamma)-O0/2*(g1+hc)
HT_0 = (pt_0**2/2+V0)*(g0**2+e0**2+1)
HT_1 = (pt_1**2/2+V1)*(g1**2+e1**2+1)

from scipy import stats
loc = (H_0+H_1)/2
#scale = d0
rvs1 = stats.norm.rvs(loc, scale = d0, size = shots)
#scale = d1
rvs2 = stats.norm.rvs(loc, scale = d1, size = shots)
stats.ttest_ind(rvs1, rvs2)
#Ttest_indResult(statistic=1.0717590970789732, pvalue=0.2840812692943359)
locT = (HT_0+HT_1)/2
#scale = d0
rvs1T = stats.norm.rvs(locT, scale = d0, size = shots)
#scale = d1
rvs2T = stats.norm.rvs(locT, scale = d1, size = shots)
stats.ttest_ind(rvs1T, rvs2T)
#Ttest_indResult(statistic=-0.3626929932163619, pvalue=0.716909226734102)

#t=2,count 0-4,x=3, top most qubit: 3/4, next qubit:6/4
