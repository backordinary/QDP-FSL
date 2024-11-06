# https://github.com/LilyHeAsamiko/QC/blob/216a52fb15464b238ca8f3903748b745af8f7682/Fast%20Quantum%20Gate/fast%20quantum%20gate%202.py
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

def operatorA(hbar, m, x, w0, p):
    assert(np.shape(x) == np.shape(p))
    hbar = 2.1091436/2 * 10**(-34)#h/pi = 2.1091436 × 10-34 m2 kg / s
    a = 1j*np.sqrt(m/2/hbar/w0)*(p/m-1j*w0*x)
    return a
    
def operatorDA(hbar, m, x, w0, p):
    assert(np.shape(x) == np.shape(p))
    hbar = 2.1091436/2 * 10**(-34)#h/pi = 2.1091436 × 10-34 m2 kg / s
    a = -1j*np.sqrt(m/2/hbar/w0)*(p/m+1j*w0*x)
    return a

def operatorP(hbar, signal): #t*x
    r,c = np.shape(signal) #cols = Dims + Labels  
    for t in range(r):
        dx = signal[:,1:c] - signal[:,0:c-1]
        P = hbar/1j*dx
    return P
    
    
hbar = 2.1091436/2 * 10**(-34)#h/pi = 2.1091436 × 10-34 m2 kg / s
#readin data
url1 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic23a1.txt'
nic23a1 = urllib.request.urlopen(url1)
#s1=requests.get(url1).content
#nic23a1=pd.read_csv(io.StringIO(s1.decode('utf-8')))
#nic23a1 = pd.read_csv(url1, delimiter='\n')
url2 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic23a3.txt'
nic23a3 = urllib.request.urlopen(url2)
#another obervations
url11 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic8a1.txt'
nic8a1 = urllib.request.urlopen(url11)
url21 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic8a3.txt'
nic8a3 = urllib.request.urlopen(url21)
Dims = 29
Labels = 8
# Do not consider the effect of time first(Spacially only)
Width = int(Dims/Labels)+1
tx1 = []
tx2 = []
#print(np.shape(nic23a1.readlines()))
for line1 in nic23a1.readlines():
    tx1.append(line1.split())
for line2 in nic23a3.readlines():
    tx2.append(line2.split())
tx1 = np.array(tx1, dtype = float) #raw: t, col: x(29)+features(8)
tx2 = np.array(tx2, dtype = float) #same data as tx1 but with different labels
rows,cols = np.shape(tx1) #cols = Dims + Labels
print(rows)
print(cols)
dataset = []
label1 = []
label2 = []
for i in range(rows):
    dataset.append(tx1[i][range(Dims)])
    label1.append(tx1[i][range(Dims,Dims+Labels)])
    label2.append(tx2[i][range(Dims,Dims+Labels)])
label1 = np.array(label1,dtype = float)
label2 = np.array(label2,dtype = float)
dataset= np.array(dataset,dtype = float)

n = 2 #{{0,1},{1,0}}   
m= 1
Pop = operatorP(hbar, dataset)
T = 100 
w0 = 2*np.pi/T
a = operatorA(hbar, m, dataset[:,1:], w0, Pop)
aD = operatorDA(hbar, m, dataset[0:np.shape(Pop)[0],0:np.shape(Pop)[1]], w0, Pop)
x2 = np.zeros((np.shape(dataset)))
x2cor = x2
x = x2
um = 1.0e-6 # MicroMeter
wavelength = 1*um
aa = a
aaD = aD

for i in range(Dims):
    temp = dataset[:,i]
    x2temp = temp
    for j in range(rows-1):        
        x2[j,i] = np.var(x2temp[0:j]**2)-np.var(np.mean(x2temp[0:j])**2)
        x[j,i] = np.mean(temp[0:j])

    
alpha = np.sqrt(2*m/hbar*x2)
dalpha = alpha
dalpha = alpha[1:np.shape(alpha)[0],:]-alpha[0:(np.shape(alpha)[0]-1),:]
velocity = x2[1:np.shape(x2)[0],:]-x2[0:(np.shape(x2)[0]-1),:]
NN = np.sqrt(np.sqrt(m/np.pi/hbar))*np.sqrt(1/wavelength) 
K = m/2/hbar/velocity
K[np.isnan(K)] = 0
K[velocity == 0] = 100* m/2/hbar
V = K
xx = K
phi = K

#simulate dynamically
for i in range(0,rows-96,T):  
    for j in range(0,Dims-1,4):
        datatemp = dataset[i:i+T,j:j+4]
        b0 = hbar/2/m/datatemp
        b0[datatemp == 0] = 100*hbar/2/m  
        if np.std(b0) < 0.05:
#        if np.std(b0) < 0.05：
            if np.mean(b0) != w0:
                coef = (b0**2-w0**2)/w0/w0
                sin2 = np.sin(w0*np.array(np.reshape(np.repeat(range(i,i+T),4),(T,4)),dtype = float))**2
                assert(np.shape(coef) == np.shape(sin2))
                post = 1+coef*sin2
                temp = x2[i:i+T,j:j+4]*post
                x2cor[i:i+T,j:j+4] = temp              #x2cor[i:i+T,j:j+4] = temp
            xx[i:i+T,j:j+4] = np.argmax(dataset[i:i+T,j:j+4],0)
            V[i:i+T,j:j+4] = m/2*w0**2*x2[i:i+T,j:j+4] 
            phi[i:i+T,j:j+4] = np.sqrt(V[i:i+T,j:j+4])
        else:
            phi[i:i+T,j:j+4] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
            aa[i:i+T,j:j+4] = operatorA(hbar, m, phi[i:i+T,j:j+4], w0, Pop[i:i+T,j:j+4])
            aaD[i:i+T,j:j+4] = operatorDA(hbar, m, phi[i:i+T,j:j+4], w0, Pop[i:i+T,j:j+4])
            V[i:i+T,j:j+4] = hbar/m*(aaD[i:i+T,j:j+4]*aa[i:i+T,j:j+4]+1/2)     
    if j == Dims-1:
        datatemp = dataset[i:i+T,j]
        b0 = hbar/2/m/datatemp
        if np.std(b0) < 0.05:
#        if np.std(b0) < 0.05：
            if np.mean(b0) != w0:
                post = 1+(b0**2-w0**2)/w0/w0*np.sin(w0*np.array(range(i,i+T),dtype = float))**2
                temp = x2[i:i+T,j]*post
                x2cor[i:i+T,j] = temp   #x2cor[i:i+T,j:j+4] = temp
                xx[i:i+T,j] = np.argmax(dataset[i:i+T,j],0)
                V[i:i+T,j] = m/2*w0**2*x2[i:i+T,j] 
                phi[i:i+T,j] = np.sqrt(V[i:i+T,j:j+4])
        else:
            phi[i:i+T,j] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
            aa[i:i+T,j] = operatorA(hbar, m, phi[i:i+T,j], w0, Pop[i:i+T,j])
            aaD[i:i+T,j] = operatorDA(hbar, m, phi[i:i+T,j], w0, Pop[i:i+T,j])
            V[i:i+T,j] = hbar/m*(aaD[i:i+T,j:j+4]*aa[i:i+T,j]+1/2)               
            

if i == rows-96:  
    for j in range(0,Dims-1,4):
        datatemp = dataset[i:rows,j:j+4]
        b0 = hbar/2/m/datatemp
        b0[datatemp == 0] = 100*hbar/2/m  
        if np.std(b0) < 0.05:
#        if np.std(b0) < 0.05：
            if np.mean(b0) != w0:
                coef = (b0**2-w0**2)/w0/w0
                sin2 = np.sin(w0*np.array(np.reshape(np.repeat(range(i,rows),4),(T,4)),dtype = float))**2
                assert(np.shape(coef) == np.shape(sin2))
                post = 1+coef*sin2
                temp = x2[i:rows,j:j+4]*post
                x2cor[i:rows,j:j+4] = temp              #x2cor[i:i+T,j:j+4] = temp
            xx[i:rows,j:j+4] = np.argmax(dataset[i:rows,j:j+4],0)
            V[i:rows,j:j+4] = m/2*w0**2*x2[i:rows,j:j+4] 
            phi[i:rows,j:j+4] = np.sqrt(V[i:rows,j:j+4])
        else:
            phi[i:rows,j:j+4] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
            aa[i:rows,j:j+4] = operatorA(hbar, m, phi[i:rows,j:j+4], w0, Pop[i:rows,j:j+4])
            aaD[i:rows,j:j+4] = operatorDA(hbar, m, phi[i:rows,j:j+4], w0, Pop[i:rows,j:j+4])
            V[i:rows,j:j+4] = hbar/m*(aaD[i:rows,j:j+4]*aa[i:rows,j:j+4]+1/2)     
    if j == Dims-1:
        datatemp = dataset[i:rows,j]
        b0 = hbar/2/m/datatemp
        if np.std(b0) < 0.05:
#        if np.std(b0) < 0.05：
            if np.mean(b0) != w0:
                post = 1+(b0**2-w0**2)/w0/w0*np.sin(w0*np.array(range(i,rows),dtype = float))**2
                temp = x2[i:rows,j]*post
                x2cor[i:rows,j] = temp   #x2cor[i:i+T,j:j+4] = temp
                xx[i:rows,j] = np.argmax(dataset[i:rows,j],0)
                V[i:rows,j] = m/2*w0**2*x2[i:rows,j] 
                phi[i:rows,j] = np.sqrt(V[i:rows,j:j+4] )
        else:
            phi[i:rows,j] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
            aa[i:rows,j] = operatorA(hbar, m, phi[i:rows,j], w0, Pop[i:rows,j])
            aaD[i:rows,j] = operatorDA(hbar, m, phi[i:rows,j], w0, Pop[i:rows,j])
            V[i:rows,j] = hbar/m*(aaD[i:i+T,j:j+4]*aa[i:rows,j]+1/2)               

       