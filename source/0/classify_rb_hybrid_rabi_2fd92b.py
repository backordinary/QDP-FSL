# https://github.com/LilyHeAsamiko/QC/blob/216a52fb15464b238ca8f3903748b745af8f7682/Grover2/classify_rb_hybrid_rabi.py
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:10:06 2020

@author: LilyHeAsamiko
"""
#Grover_puzzle_rb
#2-qubits(N = 1) oracle for w = 00, CR2 without cross-talk cancellation
#initialization
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

# center data around 0
def baseline_remove(values):
    return np.array(values) - np.mean(values)

meas_duration = 848
meas_sigma = 32
meas_square_width = 64
meas_amp = 0.2
gaussian_pulse = Gaussian(meas_duration, meas_amp, meas_sigma)
#qc1 = transpile(GC, backend)

n = 2
t = 2
N = 1
M = 4
qc = QuantumCircuit(n,t)
GC = initialize_s(qc, range(n))
GC.measure_all()
backend_defaults = backend.defaults()

#ceate the CR1 schedule

# The Unitary is an identity (with a global phase)
backend = qiskit.Aer.get_backend('unitary_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
job = qiskit.execute(qc, backend=backend, basis_gates=basis_gates)
from qiskit_textbook.tools import array_to_latex
array_to_latex(np.around(job.result().get_unitary(),3), pretext="\\text{Unitary} = ")

#Add the CR1 instruction to basis_gates and inst_map
basis_gates += ['cr1']
inst_map.add(gate_name, [1,0], sched)

#Create a quantum gate to reference the CR1 pulse schedule
cr1_gate = Gate(gate_name, 2, [])

#Create the QPT circuits
qregs = QuantumRegister(config.n_qubits)
circuit = QuantumCircuit(qregs)
circuit.append(cr1_gate, qargs = [qregs[1], qregs[0]])
qpt_circuits = process_tomography_circuits(circuit, [qregs[0], qregs[1]])

#Create the QPT pulse scchedules
qpt_circuits = transpile(qpt_circuits, backend, basis_gates)
#qpt_schedules = schedule(qpt_circuits, backend, inst_map)

#Local rotation angles for CNOT(1,0) determined during optimization+
local_rotations10 = [[1.45, 1.91, 1.64],[1.56, 3.08, 2.45],[-2.79, -3.05, -2.79],[2.16, -3.12, 0.02]]
#Local rotation angles for CNOT(1,0) determined during optimization+
local_rotations01 = [[-1.68, 3.04, 1.66],[1.57, 2.28, -0.06],[1.48, -0.46, 3.14],[1.60, -3.14, 0.98]]

#ceate the CR1 schedule
cr1_pulse = GaussianSquare(meas_duration,meas_amp, meas_sigma, meas_square_width)
sched = Schedule()
sched += Play(cr1_pulse, ControlChannel(0))

#Add the CR1 instruction to basis_gates and inst_map
basis_gates += ['cr1']
inst_map.add(gate_name, [1,0], sched)

#Create a quantum gate to reference the CR1 pulse schedule
cr1_gate = Gate(gate_name, 2, [])

#Generate RB circuits (2Q RB)
#number of qubits
nQ=2 
rb_opts = {}
#Number of Cliffords in the sequence
rb_opts['length_vector'] = [1, 51, 75, 100, 125, 150, 175, 200]
n_cliff = len(rb_opts['length_vector'])
#Number of seeds (random sequences)
rb_opts['nseeds'] = 5
n_seed = 5
#Default pattern
rb_opts['rb_pattern'] = [[0, 1]]

from qiskit.ignis.verification.randomized_benchmarking import randomized_benchmarking_seq
import qiskit.ignis.verification.randomized_benchmarking as rb
#create randomized benchmarking circuits with 5 seeds
rb_circuits_seeds, xdata = randomized_benchmarking_seq(**rb_opts)

rb_circuits_seeds[0][0].draw()

#      ┌───┐ ┌───┐ ┌───┐     ┌─────┐┌───┐ ░ ┌───┐┌───┐     ┌─────┐┌───┐┌───┐┌─┐   
#qr_0: ┤ Y ├─┤ H ├─┤ S ├──■──┤ SDG ├┤ H ├─░─┤ H ├┤ S ├──■──┤ SDG ├┤ H ├┤ Y ├┤M├───
#      ├───┤┌┴───┴┐├───┤┌─┴─┐├─────┤├───┤ ░ ├───┤├───┤┌─┴─┐└┬───┬┘├───┤├───┤└╥┘┌─┐
#qr_1: ┤ Y ├┤ SDG ├┤ H ├┤ X ├┤ SDG ├┤ H ├─░─┤ H ├┤ S ├┤ X ├─┤ H ├─┤ S ├┤ Y ├─╫─┤M├
#      └───┘└─────┘└───┘└───┘└─────┘└───┘ ░ └───┘└───┘└───┘ └───┘ └───┘└───┘ ║ └╥┘
#cr: 2/══════════════════════════════════════════════════════════════════════╩══╩═
 #                                                                           0  1 

# Create a new circuit without the measurement
qregs = rb_circuits_seeds[0][-1].qregs
cregs = rb_circuits_measseeds[0][-1].cregs
qc0 = qiskit.QuantumCircuit(*qregs, *cregs)
for i in rb_circuits_seeds[0][-1][0:-nQ]:
    qc0.data.append(i)
qregs = rb_circuits_seeds[1][-1].qregs
cregs = rb_circuits_seeds[1][-1].cregs
qc1 = qiskit.QuantumCircuit(*qregs, *cregs)
for i in rb_circuits_seeds[1][-1][0:-nQ]:
    qc1.data.append(i)    
    
# The Unitary is an identity (with a global phase)
backend = qiskit.Aer.get_backend('unitary_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
job0 = qiskit.execute(qc0, backend=backend, basis_gates=basis_gates)

from qiskit_textbook.tools import array_to_latex
array_to_latex(np.around(job0.result().get_unitary(),3), pretext="\\text{Unitary} = ")

#array([[ 3.29037480e-14+1.00000000e+00j, -4.57558474e-16-5.85837338e-16j,1.12222886e-15-8.28310237e-17j,  7.41810225e-16+7.80786793e-16j],
#       [ 2.22044605e-16+2.97521553e-17j,  3.16393136e-14+1.00000000e+00j,6.83410373e-16+3.02545808e-16j, -2.41940553e-16+2.13479482e-16j],
#       [ 1.06906579e-16+3.34767643e-17j,  6.96845055e-16-6.42299244e-16j,3.53075544e-14+1.00000000e+00j, -1.36594079e-15-2.39568580e-16j],
#       [ 3.55888035e-16-1.57646461e-15j, -4.74503714e-17-9.79138863e-16j,-1.99792283e-15+3.04518336e-16j,  3.20337549e-14+1.00000000e+00j]])

#After round
#array([[ 0.+1.j, -0.-0.j,  0.-0.j,  0.+0.j],
#       [ 0.+0.j,  0.+1.j,  0.+0.j, -0.+0.j],
#       [ 0.+0.j,  0.-0.j,  0.+1.j, -0.-0.j],
#       [ 0.-0.j, -0.-0.j, -0.+0.j,  0.+1.j]])

backend = qiskit.Aer.get_backend('unitary_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
job1 = qiskit.execute(qc1, backend=backend, basis_gates=basis_gates)

from qiskit_textbook.tools import array_to_latex
array_to_latex(np.around(job1.result().get_unitary(),3), pretext="\\text{Unitary} = ")


from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error
# Run on a noisy simulator
noise_model = NoiseModel()

# Depolarizing error on the gates u2, u3 and cx (assuming the u1 is virtual-Z gate and no error)
p1Q = 0.002
p2Q = 0.01

noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q, 1), 'u3')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')

backend = qiskit.Aer.get_backend('qasm_simulator')
job00 = qiskit.execute(qc0, backend=backend, basis_gates=basis_gates, noise_model = noise_model)
job11 = qiskit.execute(qc1, backend=backend, basis_gates=basis_gates, noise_model = noise_model)

# Create the RB fitter
basis_gates = ['u1','u2','u3','cx'] 
shots = 200
result_list0 = []
result_list1 = []
transpiled_circs_list = []
import time
rb_fit = rb.RBFitter(None, xdata, rb_opts['rb_pattern'])
rb_fit01 = rb.RBFitter(None, xdata, rb_opts['rb_pattern'])

for rb_seed, rb_circ_seed in enumerate(rb_circuits_seeds):
    print('Compiling seed %d'%rb_seed)
    new_rb_circ_seed = qiskit.compiler.transpile(rb_circ_seed, basis_gates=basis_gates)
    transpiled_circs_list.append(new_rb_circ_seed)
    print('Simulating seed %d'%rb_seed)
    job = qiskit.execute(new_rb_circ_seed, backend, shots=shots,
                         noise_model=noise_model,
                         backend_options={'max_parallel_experiments': 0})    
    # Add data to the fitter
    rb_fit.add_data(job.result())
    print('After seed %d, alpha: %f, EPC: %f'%(rb_seed,rb_fit.fit[0]['params'][1], rb_fit.fit[0]['epc']))
    for i in range(2):
        job01 = qiskit.execute(new_rb_circ_seed[i], backend=backend, shots=shots, noise_model = noise_model, backend_options={'max_parallel_experiments': 0})
        rb_fit01.add_data(job01.result())            

plt.figure()
plt.plot(rb_fit0)
           
for seed_num, data in enumerate(result_list):#range(1,len(result_list)):  
    plt.figure(figsize=(15, 6))
    axis = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
    for i in range(2):
        pattern_ind = i
        if i == 0:
            # Plot the essence by calling plot_rb_data
            rb_fit0.plot_rb_data(pattern_ind, ax=axis[i], add_label=True, show_plt=False)
            # Add title and label
            axis[i].set_title('%d Qubit RB - after seed %d'%(len(rb_opts['rb_pattern'][i]), seed_num), fontsize=18)
        else:
            rb_fit1.plot_rb_data(pattern_ind, ax=axis[i], add_label=True, show_plt=False)
            # Add title and label
            axis[i].set_title('%d Qubit RB - after seed %d'%(len(rb_opts['rb_pattern'][i]), seed_num), fontsize=18)            
    # Display
    display.display(plt.gcf())
    
    # Clear display after each seed and close
#    display.clear_output(wait=True)
#    time.sleep(1.0)
    plt.show()
#    plt.close()




#train and test
#readin data
url1 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic23a1.txt'
nic23a1 = urllib.request.urlopen(url1)
#s1=requests.get(url1).content
#nic23a1=pd.read_csv(io.StringIO(s1.decode('utf-8')))
#nic23a1 = pd.read_csv('D:/PhD in Oxford,Ethz,KI,others/OxfordPh/QC/ML(hybrid and momentum-space UCC)/nic23a1.txt')
url2 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic23a3.txt'
nic23a3 = urllib.request.urlopen(url2)
#another obervations
url21 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic8a1.txt'
nic8a1 = urllib.request.urlopen(url21)
url22 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic8a3.txt'
nic8a3 = urllib.request.urlopen(url22)

Dims = 29
Labels = 8
# Do not consider the effect of time first(Spacially only)
Width = int(Dims/Labels)+1 # particle number (batch size)
tx1 = []
tx2 = []
#print(np.shape(nic23a1.readlines()))
for line1 in nic23a1.readlines():
    tx1.append(line1.split())
for line2 in nic23a3.readlines():
    tx2.append(line2.split())
tx1 = np.array(tx1)
tx2 = np.array(tx2) #same data as tx1 but with different labels
rows,cols = np.shape(tx1) #cols = Dims + Labels
print(rows)
print(cols)
phi1 = np.ones((rows,Labels))
phi2 = np.ones((rows,Labels))
dataset = [] #896*(29+8)， consider about single person's data first(phi1 only in PSO, phi2 = 0)
label1 = []
label2 = []
for i in range(rows):
    dataset.append(tx1[i][range(Dims)])
    label1.append(tx1[i][range(Dims,Dims+Labels)])
    label2.append(tx2[i][range(Dims,Dims+Labels)])
label1 = np.array(label1,dtype = float)
label2 = np.array(label2,dtype = float)
#Tl = 100
#T = int(rows/Tl)+1

sim_result = [] # through qc classification
pred_result = [] # through classical sigmoid sgd 
train = 0.75
#trainidx = sample(range(np.shape(tx1)[0]),k = int(0.75*np.shape(tx1)[0]))
#labeltrain = label1[trainidx][:]
#testidx = sample(range(np.shape(tx1)[0]),k = int(0.25*np.shape(tx1)[0]))
#labeltest = label1[testidx][:]
Tr_Acc = np.zeros((int(np.shape(dataset)[0]*train),1))
Tr_TP = np.zeros((int(np.shape(dataset)[0]*train),1))
Tr_TN = np.zeros((int(np.shape(dataset)[0]*train),1))
Tr_FP = np.zeros((int(np.shape(dataset)[0]*train),1))
Tr_FN = np.zeros((int(np.shape(dataset)[0]*train),1))
Tr_F = np.zeros((int(np.shape(dataset)[0]*train),1))

Te_Acc = np.zeros(((np.shape(dataset)[0]-np.shape(Tr_Acc)[0]),1))
Te_TP = np.zeros(((np.shape(dataset)[0]-np.shape(Tr_Acc)[0]),1))
Te_TN = np.zeros(((np.shape(dataset)[0]-np.shape(Tr_Acc)[0]),1))
Te_FP = np.zeros(((np.shape(dataset)[0]-np.shape(Tr_Acc)[0]),1))
Te_FN = np.zeros(((np.shape(dataset)[0]-np.shape(Tr_Acc)[0]),1))
Te_F = np.zeros(((np.shape(dataset)[0]-np.shape(Tr_Acc)[0]),1))

W = np.ones((np.shape(label1)[0],np.shape(label1)[1]))
g = 2*np.log(np.sqrt(2))
#test_result = []

count = 0
p = 0.5
fs = 1200
KHz = 1000 
T = 1/fs/KHz*10**9 #us
Tl = T/cols #duratiion per lable

# cutoff dimension
dim_oscillators = 3
# frequencies for transmon drift terms
# Number of oscillators in the model is determined from len(oscillator_freqs)
oscillator_freqs = [5.0e9, 5.2e9] #harmonic term 
anharm_freqs = [-0.33e9, -0.33e9] #anharmonic term

# drive strengths
drive_strengths = [0.02e9, 0.02e9]

# specify coupling as a dictionary (qubits 0 and 1 are coupled with a coefficient 0.002e9)
coupling_dict = {(0,1): 0.002e9}

#sample duration for pulse instructions in accordance
dt = 20*backend_config.dt #2.2222222222222221e-10
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"

backend_defaults = backend.defaults()


# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
            
# We will find the qubit frequency for the following qubit.
qubit = 0

# The sweep will be centered around the estimated qubit frequency.
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz
# warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

# scale factor to remove factors of 10 from the data
scale_factor = 1e-14

# We will sweep 40 MHz around the estimated frequency
frequency_span_Hz = 40 * MHz
# in steps of 1 MHz.
frequency_step_Hz = 1 * MHz

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(frequency_min / GHz, 
                            frequency_max / GHz, 
                            frequency_step_Hz / GHz)
            
print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
      in steps of {frequency_step_Hz / MHz} MHz.")

for l in range(np.shape(label1)[0]):
    F = 0
    for i in range(np.shape(label1)[1]):
        if label1[l][i] == 1 and label2[l][i] == 1: #label1[l][i] == 1 & label2[l][i] == 1
            sim_result.append(1)
            flag = 1
        elif label1[l][i] == 0 and label2[l][i] == 0:
            sim_result.append(0)
            flag = 0
        else:
#             t = 4   # no. of counting qubits
#             n = 4   # no. of searching qubits
#             # Measure Control bits only
#             qc = QuantumCircuit(n+t, t) # Circuit with n+t qubits and t classical bits
            
#             # Initialise all qubits to |+>
#             for qubit in range(t+n):
#                 qc.h(qubit)
                
#             # Create controlled-Grover
#             gi = grover_iteration_5_16().to_gate()
#             cgi = gi.control()
#             cgi.label = "ControlGroverIteration"
            
#             # Begin controlled Grover iterations
#             iterations = 1
#             for qubit in range(t):
#                 for i in range(iterations):
#                     qc.append(cgi, [qubit] + [*range(t, n+t)])
#                 iterations *= 2           
            
#             qft_dagger = qft(4).to_gate().inverse()
#             qft_dagger.label = "QFT†"
    
#             # Do inverse QFT on counting qubits
#             qc.append(qft_dagger, range(t))
            
#             # Measure counting qubits
#             qc.measure(range(t), range(t))
#             #simulate
#             emulator = Aer.get_backend('qasm_simulator')
#             job0 = execute(qc, emulator, shots=256)
#             hist0 = job0.result().get_counts()
# #            plot_histogram(hist0)
#             #Put into EEG
#             P0 = hist0['0101']/256
#             P1 = hist0['1011']/256            
#             p = P0*0+P1*1
#             flag = round(p)

            # This experiment uses these values from the previous experiment:
                # `qubit`,
                # `measure`, and
                # `rough_qubit_frequency`.
            
            # Rabi experiment parameters
#            num_rabi_points = 50
            
            # Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
            # drive_amp_min = 0
            # drive_amp_max = 0.75
            # drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)    
 
            # # Build the Rabi experiments:
            # #    A drive pulse at the qubit frequency, followed by a measurement,
            # #    where we vary the drive amplitude each time.
            # rabi_schedules = []
            # for drive_amp in drive_amps:
            #     rabi_pulse = pulse_lib.gaussian(duration=drive_samples, amp=drive_amp, 
            #                                     sigma=drive_sigma, name=f"Rabi drive amplitude = {drive_amp}")
            #     this_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {drive_amp}")
            #     this_schedule += Play(rabi_pulse, drive_chan)
            #     # Reuse the measure instruction from the frequency sweep experiment
            #     this_schedule += measure << this_schedule.duration
            #     rabi_schedules.append(this_schedule)
       
            # Drive pulse parameters (us = microseconds)
            drive_sigma_us = Tl#0.075                     # This determines the actual width of the gaussian
            drive_samples_us = drive_sigma_us*8        # This is a truncating parameter, because gaussians don't have 
                                                       # a natural finite length
            qubits = [0,1]
            drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us /dt)       # The width of the gaussian in units of dt
            drive_samples = get_closest_multiple_of_16(drive_samples_us * us /dt)   # The truncating parameter in units of dt
            drive_amp = 0.3
            # Drive pulse samples
            drive_pulse = pulse_lib.gaussian(duration=drive_samples,
                                             sigma=drive_sigma,
                                             amp=drive_amp,
                                             name='freq_sweep_excitation_pulse')
            
            # plt.plot(drive_pulse.samples)
            # Find out which group of qubits need to be acquired with this qubit
            meas_map_idx = None
            for i, measure_group in enumerate(backend_config.meas_map):
                if qubit in measure_group:
                    meas_map_idx = i
                    break
            assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
            
            inst_sched_map = backend_defaults.instruction_schedule_map
            measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])
            
            # two_qubit_model = duffing_system_model(dim_oscillators=dim_oscillators,
            #                            oscillator_freqs=oscillator_freqs,
            #                            anharm_freqs=anharm_freqs,
            #                            drive_strengths=drive_strengths,
            #                            coupling_dict=coupling_dict,
            #                            dt=dt)
            
            ### Collect the necessary channels
            drive_chan = pulse.DriveChannel(qubit)
            meas_chan = pulse.MeasureChannel(qubit)
            acq_chan = pulse.AcquireChannel(qubit)
            
            # Create the base schedule
            # Start with drive pulse acting on the drive channel
            schedule = pulse.Schedule(name='Frequency sweep')
            schedule += Play(drive_pulse, drive_chan)
            # The left shift `<<` is special syntax meaning to shift the start time of the schedule by some duration
            schedule += measure << schedule.duration
            
            #fs = 1200
            #KHz = 1000
            # Create the frequency settings for the sweep (MUST BE IN HZ)
            #frequencies_Hz = frequencies_GHz*GHz
            frequencies_Hz = range(fs*KHz)
            schedule_frequencies = [{drive_chan: freq} for freq in frequencies_Hz]
            
            schedule.draw(label=True)
            # rabi_experiments, rabi_amps = rabi_schedules(amp_list=dataset[l][int(i*Width):int(i*Width+Width)],
            #                                  qubits=[0,1],
            #                                  pulse_width=meas_duration,
            #                                  pulse_sigma=meas_sigma,
            #                                  drives=meas_chan,
            #                                  inst_map=inst_sched_map,
            #                                  meas_map=[[0, 1]])
        # This experiment uses these values from the previous experiment:
                # `qubit`,
                # `measure`, and
                # `rough_qubit_frequency`.
            
            # Rabi experiment parameters            
            meas_amps = dataset[l][int(i*Width):int(i*Width+Width)]#np.linspace(0, 0.9, 48)
            meas_channels = [pulse.DriveChannel(0), pulse.DriveChannel(1)]
            meas_amps = np.array(meas_amps, dtype = float)
            meas_amps1 = np.linspace(min(meas_amps),max(meas_amps),len(drive_pulse.samples))

            num_rabi_points = len(drive_pulse.samples[np.array(np.linspace(0,len(drive_pulse.samples)-1,int(len(meas_amps1)/2000)),dtype = int)])
            

           
            # Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
            drive_amp_min = 0
            drive_amp_max = 0.75
            drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)    
            
            # # Build the Rabi experiments:
            # #    A drive pulse at the qubit frequency, followed by a measurement,
            # #    where we vary the drive amplitude each time.
            rabi_schedules = []
            for drive_amp in drive_amps:
                rabi_pulse = pulse_lib.gaussian(duration=drive_samples, amp=drive_amp, 
                                                sigma=drive_sigma, name=f"Rabi drive amplitude = {drive_amp}")
                this_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {drive_amp}")
                this_schedule += Play(rabi_pulse, drive_chan)
                 # Reuse the measure instruction from the frequency sweep experiment
                this_schedule += measure << this_schedule.duration
                rabi_schedules.append(this_schedule)
            # Assemble the schedules into a Qobj
            num_shots_per_point = 1024
  #          backend_sim = PulseSimulator()
#            qubit_lo_freq = two_qubit_model.hamiltonian.get_qubit_lo_from_drift()
    
            # rabi_experiments, rabi_amps = rabi_schedules(amp_list=meas_amps,
            #                                      qubits=[0,1],
            #                                      pulse_width=drive_samples_us,
            #                                      pulse_sigma=drive_sigma,
            #                                      drives=meas_channels,
            #                                      inst_map=inst_map,
            #                                      meas_map=[[0, 1]])

    
            # rabi_qobj = assemble(rabi_experiments,
            #              backend=backend,
            #              qubit_lo_freq=qubit_lo_freq,
            #              meas_level=1,
            #              meas_return='avg',
            #              shots=256)
            fit_params1, y_fit1 = fit_function(meas_amps1[np.array(list(np.linspace(0,len(drive_pulse.samples)-1,int(len(meas_amps1)/2000))),dtype = int)],
                                             drive_pulse.samples[np.array(np.linspace(0,len(drive_pulse.samples)-1,int(len(meas_amps1)/2000)),dtype = int)], #rabi_values, 
                                             lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                             [3, 0.1, 0.5, 0]) 

            A, rough_qubit_frequency, B, C = fit_params1
            rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
            print(f"We've updated our qubit frequency estimate from "
                  f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")  
                     
            rabi_experiment_program = assemble(rabi_schedules, #rabi_schedules,
                                               backend=backend,
                                               meas_level=1,
                                               meas_return='avg',
                                               shots=1, #num_shots_per_point,
                                              # qubit_lo_freq = qubit_lo_freq,
                                               #meas_lo_freq = [np.array(qubit_lo_freq)/2],
                                               #shots=num_shots_per_point,
                                               schedule_los=[{drive_chan: rough_qubit_frequency*50}]* num_rabi_points)
            
            print(job.job_id())
            
            # frequency_sweep_program = assemble(schedule,
            #                                    backend=backend, 
            #                                    meas_return='avg',
            #                                    shots=num_shots_per_frequency,
            #                                    schedule_los=schedule_frequencies)
                
            job = backend.run(rabi_experiment_program)
            job_monitor(job)
            
            rabi_results = job.result(timeout=120)
            #counts = rabi_results.get_counts()
            #p = counts/num_shots_per_pointnum_shots_per_point*
            #scale_factor = max(job.result())/drive_amp
            
            rabi_values = []
            for i in range(len(drive_pulse.samples[np.array(np.linspace(0,len(meas_amps1)-1,int(len(meas_amps1)/200)),dtype = int)])):
                # Get the results for `qubit` from the ith experiment
                rabi_values.append(rabi_results.get_memory(i)[qubit]*scale_factor)
            
            rabi_values = np.real(baseline_remove(rabi_values))
#            rabifit = RabiFitter(rabi_results, meas_amps, qubits, fit_p0 = [0.5,0.5,0.6,1.5])
#            rabifit = rb.RBFitter(None, xdata, rb_opts['rb_pattern'])
     #       meas_amps = np.array(meas_amps,dtype= float)
     #       pre_rabi_values = np.linspace(min(meas_amps),max(meas_amps),len(drive_amps))
     #       fit_params1, y_fit1 = fit_function(drive_amps,
     #                                        pre_rabi_values, 
     #                                        lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
     #                                        [3, 0.1, 0.5, 0])

        #    rabi_values = y_fit1 + np.random.normal(np.mean(y_fit1),np.std(y_fit1),len(y_fit1))

            plt.xlabel("Drive amp [a.u.]")
            plt.ylabel("Measured signal [a.u.]")
            plt.scatter(drive_pulse.samples[np.array(np.linspace(0,len(meas_amps1)-1,int(len(meas_amps1)/200)),dtype = int)], rabi_values, color='black') # plot real part of Rabi values
            plt.show()
            
            
            # fit_params1, y_fit1 = fit_function(drive_amps*sc,
            #                                  rabi_values, 
            #                                  lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
            #                                  [3, 0.1, 0.5, 0])
            
            plt.scatter(drive_pulse.samples[np.array(np.linspace(0,len(meas_amps1)-1,int(len(meas_amps1)/200)),dtype = int)], rabi_values, color='black')
            plt.plot(drive_pulse.samples[np.array(np.linspace(0,len(meas_amps1)-1,int(len(meas_amps1)/200)),dtype = int)], y_fit1, color='red')
            
            drive_period = fit_params1[2] # get period of rabi oscillation
            
            plt.axvline(drive_period/2, color='red', linestyle='--')
            plt.axvline(drive_period, color='red', linestyle='--')
            plt.annotate("", xy=(drive_period, 0), xytext=(drive_period/2,0), arrowprops=dict(arrowstyle="<->", color='red'))
            plt.annotate("$\pi$", xy=(drive_period/2-0.03, 0.1), color='red')
            
            plt.xlabel("Drive amp [a.u.]", fontsize=15)
            plt.ylabel("Measured signal [a.u.]", fontsize=15)
            plt.show()
            
            pi_amp = abs(drive_period / 2)
            print(f"Pi Amplitude = {pi_amp}")
            
            pi_pulse = pulse_lib.gaussian(duration=drive_samples,
                                          amp=pi_amp, 
                                          sigma=drive_sigma,
                                          name='pi_pulse')
            
            # Create two schedules
            
            # Ground state schedule
            gnd_schedule = pulse.Schedule(name="ground state")
            gnd_schedule += measure
            
            # Excited state schedule
            exc_schedule = pulse.Schedule(name="excited state")
            exc_schedule += Play(pi_pulse, drive_chan)  # We found this in Part 2A above
            exc_schedule += measure << exc_schedule.duration
            
            gnd_schedule.draw(label=True)
            
            exc_schedule.draw(label=True)
            
            # Execution settings
            num_shots = 1024
            
            gnd_exc_program = assemble([gnd_schedule, exc_schedule],
                                       backend=backend,
                                       meas_level=1,
                                       meas_return='single',
                                       shots=20,
                                       schedule_los=[{drive_chan: rough_qubit_frequency}] * 2)
            
            # print(job.job_id())
            job = backend.run(gnd_exc_program)
            job_monitor(job)
            
            gnd_exc_results = job.result(timeout=120)
            
            gnd_results = gnd_exc_results.get_memory(0)[:, qubit]*scale_factor
            exc_results = gnd_exc_results.get_memory(1)[:, qubit]*scale_factor
            
            plt.figure(figsize=[4,4])
            # Plot all the results
            # All results from the gnd_schedule are plotted in blue
            plt.scatter(np.real(gnd_results), np.imag(gnd_results), 
                            s=5, cmap='viridis', c='blue', alpha=0.5, label='state_0')
            # All results from the exc_schedule are plotted in red
            plt.scatter(np.real(exc_results), np.imag(exc_results), 
                            s=5, cmap='viridis', c='red', alpha=0.5, label='state_1')
            
            # Plot a large dot for the average result of the 0 and 1 states.
            mean_gnd = np.mean(gnd_results) # takes mean of both real and imaginary parts
            mean_exc = np.mean(exc_results)
            plt.scatter(np.real(mean_gnd), np.imag(mean_gnd), 
                        s=200, cmap='viridis', c='black',alpha=1.0, label='state_0_mean')
            plt.scatter(np.real(mean_exc), np.imag(mean_exc), 
                        s=200, cmap='viridis', c='black',alpha=1.0, label='state_1_mean')
            
            plt.ylabel('I [a.u.]', fontsize=15)
            plt.xlabel('Q [a.u.]', fontsize=15)
            plt.title("0-1 discrimination", fontsize=15)
            
            plt.show()
            
            if len(gnd_results) >= len(exc_results):
                sim_result.append(0)
                flag = 0
            else:
                sim_result.append(1)
                flag = 1
                
            eta = 0.1
            tol = 0.1
            #run simulation
            # consider single person data first, no pgd, use p instead
            if i < np.shape(label1)[1]:
                xtemp = np.array(dataset[l],dtype = float)
                x = xtemp[int(i*Width):int(i*Width+Width)]
                Wtemp = np.repeat(W[l][i],Width)
            else:
                xtemp = np.array(dataset[l],dtype = float)
                x = xtemp[int(-Width-1):-1]
                Wtemp = np.repeat(W[l][i],Width)                       
            for steps in range(50):
                dw = x*(1-2*label1[l][i])
                dw *= eta/len(x)
                Wtemp = Wtemp - dw
                if np.mean(abs((np.ones((np.shape(x)))+0.1)/(1+0.1-np.exp(Wtemp*x)) - label1[l][i])) < tol:
                    W[l][i] = np.mean(Wtemp)                    
                    break
            if np.mean(abs(np.ones((np.shape(x)))+0.1)/(1+0.1-np.exp(Wtemp*x)))>0.5:
                pred = 1
            else:
                pred = 0  
            pred_result.append(pred) 
        print(np.size(sim_result))
        if np.mod(l,100)<75:
            if pred_result[-1] == 1 and flag == 1:
                Tr_TP[l//100*75+np.mod(l,100)] += 1 
                Tr_Acc[l//100*75+np.mod(l,100)] += 1
            elif sim_result[-1] == 0 and flag == 1:
                Tr_TN[l//100*75+np.mod(l,100)] += 1 
            elif sim_result[-1] == 1 and flag == 0:
                Tr_FP[l//100*75+np.mod(l,100)] += 1 
            elif sim_result[-1] == 0 and flag == 0:
                Tr_FN[l//100*75+np.mod(l,100)] += 1       
                Tr_Acc[l//100*75+np.mod(l,100)] += 1
        else:
            if pred_result[-1] == 1 and flag == 1:
                Te_TP[l//100*25+np.mod(l,100)-75] += 1 
                Te_Acc[l//100*25+np.mod(l,100)-75] += 1
            elif sim_result[-1] == 0 and flag == 1:
                Te_TN[l//100*25+np.mod(l,100)-75] += 1 
            elif sim_result[-1] == 1 and flag == 0:
                Te_FP[l//100*25+np.mod(l,100)-75] += 1 
            elif sim_result[-1] == 0 and flag == 0:
                Te_FN[l//100*25+np.mod(l,100)-75] += 1       
                Te_Acc[l//100*25+np.mod(l,100)-75] += 1


# #Count the number of single and 2Q gates in the 2Q Cliffords
# gates_per_cliff = rb.rb_utils.gates_per_clifford(transpile_list,xdata[0],basis_gates,rb_opts['rb_pattern'][0])
# for basis_gate in basis_gates:
#     print("Number of %s gates per Clifford: %f "%(basis_gate ,
#                                                   np.mean([gates_per_cliff[0][basis_gate],

                                                  
                                                           
                                                           
#                                                            # Error per gate from noise model
# epgs_1q = {'u1': 0, 'u2': p1Q/2, 'u3': 2*p1Q/2}
# epg_2q = p2Q*3/4
# pred_epc = rb.rb_utils.calculate_2q_epc(
#     gate_per_cliff=gates_per_cliff,
#     epg_2q=epg_2q,
#     qubit_pair=[0, 2],
#     list_epgs_1q=[epgs_1q, epgs_1q])

# # Calculate the predicted epc
# print("Predicted 2Q Error per Clifford: %e"%pred_epc)                                                           gates_per_cliff[2][basis_gate]])))




# #Create the QPT circuits
# qregs = QuantumRegister(config.n_qubits)
# cnot10 = QuantumCircuit(qregs)
# cnot10.u3(*local_rotations10[0], qregs[0])
# cnot10.u3(*local_rotations10[1], qregs[1])
# cnot10.append(cr1_gate, qargs = [qregs[1], qregs[0]])
# cnot10.u3(*local_rotations10[2], qregs[0])
# cnot10.u3(*local_rotations10[3], qregs[1])

# cnot10 = transpile(cnot10, backend, basis_gates)

# #cnot_sched10 = schedule(cnot10, backend, inst_map)
# #cnot_sched10.draw(label = True)

# qregs = QuantumRegister(config.n_qubits)
# cnot01 = QuantumCircuit(qregs)
# cnot01.u3(*local_rotations10[0], qregs[0])
# cnot01.u3(*local_rotations10[1], qregs[1])
# cnot01.append(cr1_gate, qargs = [qregs[1], qregs[0]])
# cnot01.u3(*local_rotations10[2], qregs[0])
# cnot01.u3(*local_rotations10[3], qregs[1])

# cnot01 = transpile(cnot01, backend, basis_gates)
# #cnot_sched01 = schedule(cnot01, backend, inst_map)

# #Overwrite the default CNOT schedule in the inst_map
# #inst_map.add('cx', [1,0], cnot_sched10)
# #inst_map.add('cx', [0,1], cnot_sched01)


# #import pyfits
# #schedule the randomized branchmarking experiment into pulse schedules.
# rb_schedules_seeds = []
# #i = 0
# #plt.figure()
# for rb_circuits_seed in rb_circuits_seeds:
# #    ax = plt.subplot(len(rb_circuits_seeds),1, i+1)
# #    rb_fit.plot_rb_data(0,ax = ax)  
#     rb_circuits_seed = transpile(rb_circuits_seed, backend, basis_gates)
#     rb_schedules_seed = schedule(rb_circuits_seed, backend, inst_map)
#     rb_schedules_seeds.append(rb_schedules_seed)
# #    rb_schedules_seed[0].draw(label = True)
# #    i += 1
# #plt.show()

# # Create a new circuit without the measurement
# qr = rb_circuits_seeds[0][-1].qregs
# cr = rb_circuits_seeds[0][-1].cregs
# QC = qiskit.QuantumCircuit(*qr, *cr)
# for i in rb_circuits_seeds[0][-1][0:-n]:
#     QC.data.append(i)


# # The Unitary is an identity (with a global phase)
# backend = qiskit.Aer.get_backend('unitary_simulator')
# basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
# job = qiskit.execute(QC, backend=backend, basis_gates=basis_gates)
# from qiskit_textbook.tools import array_to_latex
# array_to_latex(np.around(job.result().get_unitary(),3), pretext="\\text{Unitary} = ")

# rabi_values = []
# for i in range(num_rabi_points):
#     # Get the results for `qubit` from the ith experiment
#     rabi_values.append(rabi_results.get_memory(i)[qubit]*scale_factor)

# rabi_values = np.real(baseline_remove(rabi_values))

# plt.xlabel("Drive amp [a.u.]")
# plt.ylabel("Measured signal [a.u.]")
# plt.scatter(drive_amps, rabi_values, color='black') # plot real part of Rabi values
# plt.show()

# pulse_schedule = schedule(GC, backend)

# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(range(8),rbfit1, color='red')
# plt.xlabel('rbfit ', fontsize=15)
# plt.ylabel('Measured Signal [a.u.]', fontsize=15)
# plt.title('QB1', fontsize=15)

# plt.subplot(1,2,2)
# plt.plot(range(5), rbfit2, color='red')
# plt.xlabel('rbfit ', fontsize=15)
# plt.ylabel('Measured Signal [a.u.]', fontsize=15)
# plt.title('QB2', fontsize=15)
# plt.show()
