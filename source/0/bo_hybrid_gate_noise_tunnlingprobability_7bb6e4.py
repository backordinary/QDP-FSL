# https://github.com/LilyHeAsamiko/QC/blob/216a52fb15464b238ca8f3903748b745af8f7682/hyybrid%2BVQE/BO_hybrid_gate_noise_tunnlingprobability.py
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 06:25:53 2020

@author: LilyHeAsamiko
"""

#%matplotlib inline
#The model is specified in terms of the following parameters:
#Each Duffing oscillator is specified by a frequency ν, anharmonicity α, and drive strength r, which result in the Hamiltonian terms:

#2πνa†a+παa†a(a†a−1)+2πr(a+a†)×D(t),
#where D(t) is the signal on the drive channel for the qubit, and a† and a are, respectively, the creation and annihilation operators for the qubit. Note that the drive strength r sets the scaling of the control term, with D(t) assumed to be a complex and unitless number satisfying |D(t)|≤1. - A coupling between a pair of oscillators (l,k) is specified by the coupling strength J, resulting in an exchange coupling term:

#2πJ(ala†k+a†lak),
#where the subscript denotes which qubit the operators act on. - Additionally, for numerical simulation, it is necessary to specify a cutoff dimension; the Duffing oscillator model is infinite dimensional, and computer simulation requires restriction of the operators to a finite dimensional subspace.


#%matplotlib inline
import qiskit
#qiskit.__version__
from qiskit import IBMQ
MY_API_TOKEN = 'a93830f80226030329fc4e2e4d78c06bdf1942ce349fcf8f5c8021cfe8bd5abb01e4205fbd7b9c34f0b26bd335de7f1bcb9a9187a2238388106d16c6672abea2'
provider = IBMQ.enable_account(MY_API_TOKEN)
IBMQ.save_account(MY_API_TOKEN , overwrite=True)
#IBMQ.get_provider(hub='ibm-q', group='open', project='main')
#IBMQ.load_account()
print(IBMQ.providers())
backend = provider.get_backend('ibmq_qasm_simulator')
#IBMQ.update_account(MY_API_TOKEN)
#provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
#backend = provider.get_backend('ibmq_armonk')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
#provider = IBMQ.load_account()
import warnings
warnings.filterwarnings('ignore')
from qiskit.tools.jupyter import *
#backend_config = backend.configuration(a93830f80226030329fc4e2e4d78c06bdf1942ce349fcf8f5c8021cfe8bd5abb01e4205fbd7b9c34f0b26bd335de7f1bcb9a9187a2238388106d16c6672abea2)
#assert backend_config.open_pulse, "Backend doesn't support Pulse"

from numpy import *
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

import os
import sys
import io
import requests
import urllib
#store pi amplitudes
#given the drive and target indices, and the option to either start with the drive qubit in the ground or excited state, returns a list of experiments for observing the oscillations.
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
import pandas as pd

# Import Qiskit classes
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.providers.aer import noise
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error
#Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb
import time

def cr_drive_experiments(drive_idx,
                         target_idx,
                         flip_drive_qubit = False,

                         #cr_drive_amps=np.linspace(0, 0.9, 16),
                         #cr_drive_samples=800,
                         #cr_drive_sigma=4,
                         #pi_drive_samples=128,
                         #pi_drive_sigma=16
                         #meas_amp = Dims/128*0.025/2
                         #meas_width = int(rows/128*1150/2)
                         cr_drive_amps=np.linspace(0, 0.9, meas_width**2),
                         cr_drive_samples=sum(label1[:] == label2[:]),
                         cr_drive_sigma=meas_sigma,
                         pi_drive_samples=sum((label1[:] ==1)*(label2[:] ==1)), #label1[:] ==1 and label2[:] ==1
                         pi_drive_sigma=cr_drive_sigma**2):
    """Generate schedules corresponding to CR drive experiments.

    Args:
        drive_idx (int): label of driven qubit
        target_idx (int): label of target qubit
        flip_drive_qubit (bool): whether or not to start the driven qubit in the ground or excited state
        cr_drive_amps (array): list of drive amplitudes to use
        cr_drive_samples (int): number samples for each CR drive signal
        cr_drive_sigma (float): standard deviation of CR Gaussian pulse
        pi_drive_samples (int): number samples for pi pulse on drive
        pi_drive_sigma (float): standard deviation of Gaussian pi pulse on drive

    Returns:
        list[Schedule]: A list of Schedule objects for each experiment
    """

    # Construct measurement commands to be used for all schedules
    #meas_amp = 0.025
    #meas_samples = 1200
    #meas_sigma = 4
    #meas_width = 1150
    meas_amp = 0.025
    meas_sigma = 4
    ni = int(np.ceil(cols/meas_sigma))
    print(ni)
    meas_samples = rows*ni
    meas_width = int(rows*ni*23/24)
    meas_pulse = GaussianSquare(duration=meas_samples, amp=meas_amp/np.linalg.norm(meas_amp),
                               sigma=meas_sigma, width=meas_width)

    acq_sched = pulse.Acquire(meas_samples, pulse.AcquireChannel(0), pulse.MemorySlot(0))
    acq_sched += pulse.Acquire(meas_samples, pulse.AcquireChannel(1), pulse.MemorySlot(1))

    # create measurement schedule
    measure_sched = (pulse.Play(meas_pulse, pulse.MeasureChannel(0)) |
                     pulse.Play(meas_pulse, pulse.MeasureChannel(1))|
                     acq_sched)

    # Create schedule
    schedules = []
    for ii, cr_drive_amp in enumerate(cr_drive_amps):

        # pulse for flipping drive qubit if desired
        pi_pulse = Gaussian(duration=pi_drive_samples, amp=pi_amps[drive_idx], sigma=pi_drive_sigma)

        # cr drive pulse
        cr_width = cr_drive_samples - 2*cr_drive_sigma*4
        cr_rabi_pulse = GaussianSquare(duration=cr_drive_samples,
                                       amp=cr_drive_amp/np.linalg.norm(cr_drive_amp),
                                       sigma=cr_drive_sigma,
                                       width=cr_width)

        # add commands to schedule
        schedule = pulse.Schedule(name='cr_rabi_exp_amp_%s' % cr_drive_amp)
        #schedule = pulse.Schedule(name='cr_rabi_exp_amp_%s' % cr_drive_amp/np.linalg.norm(cr_drive_amp))

        # flip drive qubit if desired
        if flip_drive_qubit:
            schedule += pulse.Play(pi_pulse, pulse.DriveChannel(drive_idx))

        # do cr drive
        # First, get the ControlChannel index for CR drive from drive to target
        cr_idx = two_qubit_model.control_channel_index((drive_idx, target_idx))
        schedule += pulse.Play(cr_rabi_pulse, pulse.ControlChannel(cr_idx))  << schedule.duration


        schedule += measure_sched << schedule.duration

        schedules.append(schedule)
    return schedules

#create two functions for observing the data: 
#- plot_cr_pop_data - for plotting the oscillations between the ground state and the first excited state
#- plot_bloch_sphere - for viewing the trajectory of the target qubit on the bloch sphere
def plot_cr_pop_data(drive_idx,
                     target_idx,
                     sim_result,
                     cr_drive_amps=np.linspace(0, 0.9, 16)):
#                     cr_drive_amps=np.linspace(0, 0.9, Dims)):
#                     cr_drive_amps=cr_drive_amp/np.linalg.norm(cr_drive_amp)):
    """Plot the population of each qubit.

    Args:
        drive_idx (int): label of driven qubit
        target_idx (int): label of target qubit
        sim_result (Result): results of simulation
        cr_drive_amps (array): list of drive amplitudes to use for axis labels
    """

    amp_data_Q0 = []
    amp_data_Q1 = []

    for exp_idx in range(len(cr_drive_amps)):
        exp_mem = sim_result.get_memory(exp_idx)
        amp_data_Q0.append(np.abs(exp_mem[0]))
        amp_data_Q1.append(np.abs(exp_mem[1]))

    plt.plot(cr_drive_amps, amp_data_Q0, label='Q0')
    plt.plot(cr_drive_amps, amp_data_Q1, label='Q1')
    plt.legend()
    plt.xlabel('Pulse amplitude, a.u.', fontsize=20)
    plt.ylabel('Signal, a.u.', fontsize=20)
    plt.title('CR (Target Q{0}, driving on Q{1})'.format(target_idx, drive_idx), fontsize=20)
    plt.grid(True)

def bloch_vectors(drive_idx, drive_energy_level, sim_result):
    """Plot the population of each qubit.

    Args:
        drive_idx (int): label of driven qubit
        drive_energy_level (int): energy level of drive qubit at start of CR drive
        sim_result (Result): results of simulation

    Returns:
        list: list of Bloch vectors corresponding to the final state of the target qubit
              for each experiment
    """

    # get the dimension used for simulation
    dim = int(np.sqrt(len(sim_result.get_statevector(0))))


    # get the relevant dressed state indices
    idx0 = 0
    idx1 = 0
    if drive_idx == 0:
        if drive_energy_level == 0:
            idx0, idx1 = 0, dim
        elif drive_energy_level == 1:
            idx0, idx1 = 1, dim + 1
    if drive_idx == 1:
        if drive_energy_level == 0:
            idx0, idx1 = 0, 1
        elif drive_energy_level == 1:
            idx0, idx1 = dim, dim + 1

    # construct Pauli operators for correct dressed manifold
    state0 = np.array([two_qubit_model.hamiltonian._estates[idx0]])
    state1 = np.array([two_qubit_model.hamiltonian._estates[idx1]])

    outer01 = np.transpose(state0)@state1
    outer10 = np.transpose(state1)@state0
    outer00 = np.transpose(state0)@state0
    outer11 = np.transpose(state1)@state1

    X = outer01 + outer10
    Y = -1j*outer01 + 1j*outer10
    Z = outer00 - outer11

    # function for computing a single bloch vector
    bloch_vec = lambda vec: np.real(np.array([np.conj(vec)@X@vec, np.conj(vec)@Y@vec, np.conj(vec)@Z@vec]))

    return [bloch_vec(sim_result.get_statevector(idx)) for idx in range(len(sim_result.results))]

def plot_bloch_sphere(bloch_vectors):
    """Given a list of Bloch vectors, plot them on the Bloch sphere

    Args:
        bloch_vectors (list): list of bloch vectors
    """
    sphere = Bloch()
    sphere.add_points(np.transpose(bloch_vectors))
    sphere.show()

def exp2target(y):
    return np.ones((np.size(y)))/(1+np.exp(y))

def KL(p,q):#p:used distribution for data, q: designed distribution for model
    return -sum(p*np.log(q/p))

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
Width = int(Dims/Labels)+1
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
dataset = []
label1 = []
label2 = []
for i in range(rows):
    dataset.append(tx1[i][range(Dims)])
    label1.append(tx1[i][range(Dims,Dims+Labels)])
    label2.append(tx2[i][range(Dims,Dims+Labels)])
label1 = np.array(label1,dtype = float)
label2 = np.array(label2,dtype = float)

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
dt = 1e-9 #backend_config.dt 

# create the Duffing oscillator system model(returns a PulseSystemModel object, which is a general object for storing model information required for simulation with the PulseSimulator.)
two_qubit_model = duffing_system_model(dim_oscillators=dim_oscillators,
                                       oscillator_freqs=oscillator_freqs,
                                       anharm_freqs=anharm_freqs,
                                       drive_strengths=drive_strengths,
                                       coupling_dict=coupling_dict,
                                       dt=dt)

#calibrate pi pulse on each qubit using Ihnis(default GaussianSquare)
#4.1 Constructing the schedules
# list of qubits to be used throughout the notebook

qubits = [0, 1]
# Construct a measurement schedule and add it to an InstructionScheduleMap
meas_amp = 0.025
meas_sigma = 4
ni = int(np.ceil(cols/meas_sigma))
print(ni)
meas_samples = rows*ni
meas_width = int(rows*ni*23/24)

meas_pulse = GaussianSquare(duration=meas_samples, amp=meas_amp,
                            sigma=meas_sigma, width=meas_width)

acq_sched = pulse.Acquire(meas_samples, pulse.AcquireChannel(0), pulse.MemorySlot(0))
acq_sched += pulse.Acquire(meas_samples, pulse.AcquireChannel(1), pulse.MemorySlot(1))

measure_sched = pulse.Play(meas_pulse, pulse.MeasureChannel(0)) | pulse.Play(meas_pulse, pulse.MeasureChannel(1)) | acq_sched

inst_map = pulse.InstructionScheduleMap()
inst_map.add('measure', qubits, measure_sched)

#Rabi schedules
#recall: Rabii oscillation
# The magnetic moment is thus {\displaystyle {\boldsymbol {\mu }}={\frac {\hbar }{2}}\gamma {\boldsymbol {\sigma }}}{\boldsymbol {\mu }}={\frac {\hbar }{2}}\gamma {\boldsymbol {\sigma }}. 
# The Hamiltonian of this system is then given by {H} =-{{\mu }}\cdot{B} =-{\frac {\hbar }{2}}\omega _{0}\sigma _{z}-{\frac {\hbar }{2}}\omega _{1}(\sigma _{x}\cos \omega t-\sigma _{y}\sin \omega t)}\mathbf {H} =-{\boldsymbol {\mu }}\cdot \mathbf {B} =-{\frac {\hbar }{2}}\omega _{0}\sigma _{z}-{\frac {\hbar }{2}}\omega _{1}(\sigma _{x}\cos \omega t-\sigma _{y}\sin \omega t) where {\displaystyle \omega _{0}=\gamma B_{0}}\omega _{0}=\gamma B_{0} and {\displaystyle \omega _{1}=\gamma B_{1}}\omega _{1}=\gamma B_{1}
# Now, let the qubit be in state {\displaystyle |0\rangle }{\displaystyle |0\rangle } at time {\displaystyle t=0}t=0. Then, at time {\displaystyle t}t, the probability of it being found in state {\displaystyle |1\rangle }|1\rangle  is given by {\displaystyle P_{0\to 1}(t)=\left({\frac {\omega _{1}}{\Omega }}\right)^{2}\sin ^{2}\left({\frac {\Omega t}{2}}\right)}{\displaystyle P_{0\to 1}(t)=\left({\frac {\omega _{1}}{\Omega }}\right)^{2}\sin ^{2}\left({\frac {\Omega t}{2}}\right)} where {\displaystyle \Omega ={\sqrt {(\omega -\omega _{0})^{2}+\omega _{1}^{2}}}}\Omega ={\sqrt {(\omega -\omega _{0})^{2}+\omega _{1}^{2}}}
# the qubit oscillates between the {\displaystyle |0\rangle }|0\rang and {\displaystyle |1\rangle }|1\rangle  states. 
# The maximum amplitude for oscillation is achieved at {\displaystyle \omega =\omega _{0}}\omega =\omega _{0}, which is the condition for resonance. 
# At resonance, the transition probability is given by {\displaystyle P_{0\to 1}(t)=\sin ^{2}\left({\frac {\omega _{1}t}{2}}\right)}{\displaystyle P_{0\to 1}(t)=\sin ^{2}\left({\frac {\omega _{1}t}{2}}\right)}

# pi pulse: 
# To go from state {\displaystyle |0\rangle }|0\rang to state {\displaystyle |1\rangle }|1\rangle  it is sufficient to adjust the time {\displaystyle t}t during which the rotating field acts such that {\displaystyle {\frac {\omega _{1}t}{2}}={\frac {\pi }{2}}}{\frac {\omega _{1}t}{2}}={\frac {\pi }{2}} or {\displaystyle t={\frac {\pi }{\omega _{1}}}}t={\frac {\pi }{\omega _{1}}}
#  If a time intermediate between 0 and {\displaystyle {\frac {\pi }{\omega _{1}}}}{\frac {\pi }{\omega _{1}}} is chosen, we obtain a superposition of {\displaystyle |0\rangle }|0\rang and {\displaystyle |1\rangle }|1\rangle . 
# Approximation with pi/2 pulse: 
# In particular for {\displaystyle t={\frac {\pi }{2\omega _{1}}}}t={\frac {\pi }{2\omega _{1}}}, we have a {\displaystyle {\frac {\pi }{2}}}{\frac {\pi }{2}} pulse, which acts as: {\displaystyle |0\rangle \to {\frac {|0\rangle +i|1\rangle }{\sqrt {2}}}}{\displaystyle |0\rangle \to {\frac {|0\rangle +i|1\rangle }{\sqrt {2}}}}.
# The equations are essentially identical in the case of a two level atom in the field of a laser when the generally well satisfied rotating wave approximation is made. Then {\displaystyle \hbar \omega _{0}}\hbar \omega _{0} is the energy difference between the two atomic levels, {\displaystyle \omega }\omega  is the frequency of laser wave and Rabi frequency {\displaystyle \omega _{1}}\omega _{1} is proportional to the product of the transition electric dipole moment of atom {\displaystyle {\vec {d}}}{\vec {d}} and electric field {\displaystyle {\vec {E}}}{\vec {E}} of the laser wave that is {\displaystyle \omega _{1}\propto \hbar \ {\vec {d}}\cdot {\vec {E}}}{\displaystyle \omega _{1}\propto \hbar \ {\vec {d}}\cdot {\vec {E}}}.

# pre-run the models for 
# experiments 
drive_amps = np.linspace(0, 0.9, 3*meas_sigma)
drive_sigma = meas_sigma**2
drive_duration = int(8*drive_sigma)

drive_channels = [pulse.DriveChannel(0), pulse.DriveChannel(1)]

rabi_experiments0, rabi_amps0 = rabi_schedules(amp_list=drive_amps,
                                             qubits=qubits,
                                             pulse_width=drive_duration,
                                             pulse_sigma=drive_sigma,
                                             drives=drive_channels,
                                             inst_map=inst_map,
                                             meas_map=[[0, 1]])

rabi_experiments0[10].draw()

#ground_freq_sweep_job = backend.run(ground_freq_sweep_program)
#print(ground_freq_sweep_job.job_id())
#job_monitor(ground_freq_sweep_job)
# Get the job data (average)
#ground_freq_sweep_data = get_job_data(ground_freq_sweep_job, average=True)
#To simulate the Rabi experiments, assemble the Schedule list into a qobj. When assembling, pass the PulseSimulator as the backend.#To simulate the Rabi experiments, assemble the Schedule list into a qobj. When assembling, pass the PulseSimulator as the backend.

# instantiate the pulse simulator
#backend_sim = PulseSimulator()
backend_sim = qiskit.providers.aer.PulseSimulator()

# compute frequencies from the Hamiltonian
qubit_lo_freq0 = two_qubit_model.hamiltonian.get_qubit_lo_from_drift()

rabi_qobjlo0 = assemble(rabi_experiments0,
                     backend=backend_sim,
                     qubit_lo_freq=qubit_lo_freq0,
                     meas_level=1,
                     meas_return='avg',
                     shots=256)

# run the simulation
rabi_resultlo0 = backend_sim.run(rabi_qobjlo0, two_qubit_model).result()

rabifitlo0 = RabiFitter(rabi_resultlo0, rabi_amps0, qubits, fit_p0 = [0.5,0.5,0.6,1.5])

plt.figure(figsize=(15, 10))
q_offset = 0
multiplier = 0.5
for qubit in qubits:
#    ax = plt.subplot(2, 2, qubit + 1)
    #Xvmin, Xvmax = ax.xaxis.get_data_interval()
    #Yvmin, Yvmax = ax.yaxis.get_data_interval()
    #print(Xvmin, Xvmax,Yvmin, Yvmax)
#    Xvmin = multiplier * np.floor(0.1 / multiplier)
#    Xvmax = multiplier * np.ceil(0.5 / multiplier)
#    ax.set_xlim([Xvmin, Xvmax])
#    rabifitlo0.plot(qubit, ax=ax)
    print('Pi Amp: %f'%rabifitlo0.pi_amplitude(qubit))
plt.show()


pi_amps = [rabifitlo0.pi_amplitude(0), rabifitlo0.pi_amplitude(1)]


# construct experiments to observe CR oscillations on qubit 0, driving qubit 1
# Qubit 1（and qubit 0000000000） in the ground state

# construct experiments 
cr_drive_sigma = meas_sigma
#drive_idx = 1
#target_idx = 0
#flip_drive = False
#experiments10 = cr_drive_experiments(drive_idx, target_idx, flip_drive)

# compute frequencies from the Hamiltonian
#qubit_lo_freq10 = two_qubit_model.hamiltonian.get_qubit_lo_from_drift()

# assemble the qobj0
#cr_rabi_qobj0 = assemble(experiments10,
#                        backend=backend_sim,
#                        qubit_lo_freq=qubit_lo_freq10,
#                        meas_level=1,
#                        meas_return='avg',
#                        shots=256)

#simulation
#sim_result1= backend_sim.run(cr_rabi_qobj0, two_qubit_model).result()

#plot_cr_pop_data(drive_idx, target_idx, sim_result1)

# observe the trajectory of qubit 0 on the Bloch sphere:
#bloch_vecs1 = bloch_vectors(drive_idx, int(flip_drive), sim_result0)
#print(np.mean(bloch_vecs0*N))
#print(N)
#if (np.mean(bloch_vecs0*N) < 1):
#    plot_bloch_sphere(bloch_vecs0*N)
#else:
#    plot_bloch_sphere(bloch_vecs0)

# flag = 1
experiments1, amps1 = rabi_schedules(amp_list=drive_amps,
                                             qubits=qubits,
                                             pulse_width=drive_duration,
                                             pulse_sigma=drive_sigma,
                                             drives=drive_channels,
                                             inst_map=inst_map,
                                             meas_map=[[0, 1]])
qubit_lo_freq1 = two_qubit_model.hamiltonian.get_qubit_lo_from_drift()
crabi_qobjlo1 = assemble(experiments1,
                         backend=backend_sim,
                         qubit_lo_freq=qubit_lo_freq1,
                         meas_level=1,
                         meas_return='avg',
                         noise_model = None, #NoiseModel(),
                         shots=256)
result1 = backend_sim.run(crabi_qobjlo1, two_qubit_model).result()

 
# flag = 0
experiments0, amps0 = rabi_schedules(amp_list=drive_amps,
                                     qubits=qubits,
                                     pulse_width=drive_duration,
                                     pulse_sigma=drive_sigma,
                                     drives=drive_channels,
                                     inst_map=inst_map,
                                     meas_map=[[1, 0]])
qubit_lo_freq0 = two_qubit_model.hamiltonian.get_qubit_lo_from_drift()
crabi_qobjlo0 = assemble(experiments0,
                         backend=backend_sim,
                         qubit_lo_freq=qubit_lo_freq0,
                         meas_level=1,
                         meas_return='avg',
                         noise_model = None, #NoiseModel(),
                         shots=256)
result0 = backend_sim.run(crabi_qobjlo0, two_qubit_model).result()


sim_result = []
train = 0.75
trainidx = sample(range(np.shape(tx1)[0]),k = int(0.75*np.shape(tx1)[0]))
labeltrain = label1[trainidx][:]
# random sampling 
# testidx = sample(range(np.shape(tx1)[0]),k = int(0.25*np.shape(tx1)[0]))
# cross-validation
testidx = []
for i in range(np.shape(tx1)[0]):
    if i not in trainidx: 
        testidx.append(i)
labeltest = label1[testidx][:]
W = ones((np.shape(labeltrain)[1],meas_sigma*3))
test_result = []
# each signal as two labels marked by two different physicians.
# consider VQE on each signal contains two qubits and the 0 is 0, 1 is 1. The reuslt is the higher possibility one.(Each signal is independent.)

#Number of qubits
nQ = 3
#There are 3 qubits: Q0,Q1,Q2.
#Number of seeds (random sequences)
nseeds = 5
#Number of Cliffords in the sequence (start, stop, steps)
nCliffs = np.arange(1,200,20)
#2Q RB on Q0,Q2 and 1Q RB on Q1
rb_pattern = [[0,1],[2]]
#Do three times as many 1Q Cliffords
length_multiplier = [1,3]

#Generate the RB sequences
rb_opts = {}
rb_opts['length_vector'] = nCliffs
rb_opts['nseeds'] = nseeds
rb_opts['rb_pattern'] = rb_pattern
rb_opts['length_multiplier'] = length_multiplier
rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)

#The H is an arbitary state (with a global phase)
#The U is an identity (with a global phase)
backend = qiskit.Aer.get_backend('qasm_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX 
#print(np.around(job.result().get_unitary(), 3))    

#define the noise model
noise_model = NoiseModel()
p1Q = 0.002
p2Q = 0.01
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
noise_model.add_all_qubit_quantum_error(depolarizing_error(2*p1Q, 1), 'u3')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')

backend = qiskit.Aer.get_backend('qasm_simulator')
shots = 100
result_list = []
transpile_list = []
E = []
rb_fit = []

for l in trainidx:
    for i in range(np.shape(labeltrain)[1]):
        if label1[l][i] == 1 and label2[l][i] == 1: #label1[l][i] == 1 & label2[l][i] == 1
            sim_result.append(1)               
        elif label1[l][i] == 0 and label2[l][i] == 0:
            sim_result.append(0)
        else:
            qr = qiskit.QuantumRegister(2)
#            cr = qiskit.ClassicalRegister(1)   
#            qc = QuantumCircuit(qr, cr)
            qc = QuantumCircuit(qr)                  
            qc.h(qr[0])
            qc.cx(qr[0],qr[1])
            qc.h(qr[0])
            qubit_list = [0,1]
            
            backend = qiskit.Aer.get_backend('unitary_simulator')
            
            qc2 = qiskit.QuantumCircuit(*rb_circs[0][-1].qregs,*rb_circs[0][-1].cregs)
            #Add rb to previous circuit
            for ii in rb_circs[0][-1][0:-nQ]:
                 qc2.data.append(ii)
    
            backend = qiskit.Aer.get_backend('qasm_simulator')
            basis_gates = ['u1','u2','u3','cx'] # use U,CX 
            #print(np.around(job.result().get_unitary(), 3))    
            
            #define the noise model
            noise_model = NoiseModel()
            p1Q = 0.002
            p2Q = 0.01
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
            noise_model.add_all_qubit_quantum_error(depolarizing_error(2*p1Q, 1), 'u3')
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')
            
            shots = 100
            result_list = []
            transpile_list = []
            
            plt.figure(figsize=(15, 6))
            qc2.draw()
            # after Seed 1
            for i in range(2):
                ax = plt.subplot(1, 2, i+1)
                pattern_ind = i
                # Plot the essence by calling plot_rb_data
 #               rbfit.plot_rb_data(pattern_ind, ax=ax, add_label=True, show_plt=False)
                
                # Add title and label
                ax.set_title('%d Qubit RB'%(len(rb_opts['rb_pattern'][i])), fontsize=18)
            plt.show()                    
        
            if label1[l][i] == 1 and label2[l][i] == 0:
                drive_idx = 1
                target_idx = 0 
                flip_drive = True
                qresult = qc.measure_all()
                backend_sim = qiskit.providers.aer.PulseSimulator()
#                experiments10 = cr_drive_experiments(drive_idx, target_idx, flip_drive)
               # compute frequencies from the Hamiltonian
#                qubit_lo_freq10 = two_qubit_model.hamiltonian.get_qubit_lo_from_drift()
                #assemble qobj 1
                rabifit_result = RabiFitter(result1, amps1, qubits, fit_p0 = [0.5,0.5,0.6,1.5])
                if i == 0:
                    qc.draw()  
#                    
#        ┌───┐      ░ ┌─┐   
#  q2_0: ┤ H ├──■───░─┤M├───
#        └───┘┌─┴─┐ ░ └╥┘┌─┐
#  q2_1: ─────┤ X ├─░──╫─┤M├
#             └───┘ ░  ║ └╥┘
#meas: 2/══════════════╩══╩═
#                           0  1 
#                    plt.figure(figsize=(15, 10))
                    for qubit in qubits:
     #                   ax = plt.subplot(2, 2, qubit + 1)
                        #Xvmin, Xvmax = ax.xaxis.get_data_interval()
                        #Yvmin, Yvmax = ax.yaxis.get_data_interval()
                        #print(Xvmin, Xvmax,Yvmin, Yvmax)
     #                   Xvmin = multiplier * np.floor(0.1 / multiplier)
     #                   Xvmax = multiplier * np.ceil(0.5 / multiplier)
     #                   ax.set_xlim([Xvmin, Xvmax])
#                        rabifit_result.plot(qubit, ax=ax)
                        print('Q1 Pi Amp: %f'%rabifit_result.pi_amplitude(qubit))
                flag = 1

            if label1[l][i] == 0 and label2[l][i] == 1:
                drive_idx = 0
                target_idx = 1 
                flip_drive = True
                qresult = qc.measure_all()
                backend_sim = qiskit.providers.aer.PulseSimulator()
#                qubit_lo_freq01 = two_qubit_model.hamiltonian.get_qubit_lo_from_drift()
#                experiments10 = cr_drive_experiments(drive_idx, target_idx, flip_drive)
               # compute frequencies from the Hamiltonian
                rabifit_result = RabiFitter(result0, amps0, qubits, fit_p0 = [0.5,0.5,0.6,1.5])
                if i == 0:
                    qc.draw()  
#                    plt.figure(figsize=(15, 10))
                    for qubit in qubits:
 #                       ax = plt.subplot(2, 2, qubit + 1)
                        #Xvmin, Xvmax = ax.xaxis.get_data_interval()
                        #Yvmin, Yvmax = ax.yaxis.get_data_interval()
                        #print(Xvmin, Xvmax,Yvmin, Yvmax)
#                        Xvmin = multiplier * np.floor(0.1 / multiplier)
#                        Xvmax = multiplier * np.ceil(0.5 / multiplier)
#                       ax.set_xlim([Xvmin, Xvmax])
#                       rabifit_result.plot(qubit, ax=ax)
                        print('Q0 Pi Amp: %f'%rabifit_result.pi_amplitude(qubit))
                flag = 0   
                  
            eta = 0.1
            tol = 0.1
            #run simulation
            q_offset = 0
            multiplier = 0.5
            
 #           fitseed = []
 #           teempdata = []
 #            rbfit = rb.fitters.RBFitter(result_list[0], xdata, rb_opts['rb_pattern'])
 #            for seed_num, data in enumerate(result_list):#range(1,len(result_list)):
 # #               fitseed.append(seed_num)
 # #               tempdata.append(data)
 #                plt.figure(figsize=(15, 6))
 #                axis = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
            
 #                # Add another seed to the data
 #                rbfit.add_data([data])
                
 #                for i in range(2):
 #                    pattern_ind = i
                
 #                    # Plot the essence by calling plot_rb_data
 #                    rbfit.plot_rb_data(pattern_ind, ax=ax, add_label=True, show_plt=False)
                
 #                    # Add title and label
 #                    axis[i].set_title('%d Qubit RB - after seed %d'%(len(rb_opts['rb_pattern'][i]), seed_num), fontsize=18)
                
 #                # Display
 #                display.display(plt.gcf())
                    
 #                # Clear display after each seed and close
 #                display.clear_output(wait=True)
 #                time.sleep(1.0)
 #                plt.close()
            shots = 200
            result_list = []
            transpile_list = []
            if i == 0:
                for rb_seed,rb_circ_seed in enumerate(rb_circs):
                    print('Compiling seed %d'%rb_seed)
                    rb_circ_transpile = qiskit.transpile(rb_circ_seed, basis_gates=basis_gates)
                    print('Simulating seed %d'%rb_seed)
                    job = qiskit.execute(rb_circ_transpile, noise_model=noise_model, shots=shots, backend=backend, backend_options={'max_parallel_experiments': 0})
                    result_list.append(job.result())
                    transpile_list.append(rb_circ_transpile)
                    print([l,i])
            print("Finished Simulating")

            #Add this data to the previous fit
#            rbfit.add_data(result_list)
            
            #Replot
            plt.figure(figsize=(15, 6))
            
            for i in range(2):
                ax = plt.subplot(1, 2, i+1)
                pattern_ind = i
            
                rb_fit = rb.RBFitter(None, xdata, rb_opts['rb_pattern'])
                # Plot the essence by calling plot_rb_data
 #               rbfit.plot_rb_data(pattern_ind, ax=ax, add_label=True, show_plt=True)
            
                # Add title and label
                ax.set_title('%d Qubit RB'%(len(rb_opts['rb_pattern'][i])), fontsize=18)
            
            plt.show()
            
#            cr_result = rabifit_result.ydata['0'][flag]['mean']+np.random.normal(0,1,size = np.shape(rabifit_result.ydata['0'][flag]['mean']))/10
#            width = len(cr_result)
            width = np.shape(labeltrain)[1]#xdata
#            if i < np.shape(labeltrain)[1]:
            if i < 8:
                x = np.array(dataset[l][int(i*width/6):int(i*width/6+width)],dtype = float)
                Wtemp = W[i][:]
            else:
                x = np.array(dataset[l][int(-width-1):-1],dtype = float)
#                cr_result = cr_result[int(-width-1):-1]
                Wtemp = W[i][int(-width-1):-1]
                            
            for steps in range(50):
#                dw = cr_result/np.linalg.norm(cr_result)+x*(1-2*label1[l][i])
#                dw *= eta/len(x)
                dw = eta/len(x)
                Wtemp = Wtemp - dw
                temprst = abs(ones((np.shape(x)))/(1-np.exp(dot(Wtemp[0:len(x)],x))))
                if np.mean(temprst- label1[l][i]) < tol:
                    if i < np.shape(labeltrain)[1]:
                        W[i][:] = Wtemp
                    else:
                        W[i][int(-width-1):-1] = Wtemp                        
                    break
            if np.mean(temprst) > 0.5:
                pred = 1
            else:
                pred = 0
            sim_result.append(pred)
#sim_result = np.reshape(np.array(sim_result),(np.shape(labeltrain)))
sim_result1 = sim_result
sim_result = np.reshape(np.array(sim_result1[0:10496]),(1312,8))
#accuracy for training
Tr_Acc = np.zeros((np.shape(labeltrain)))
Tr_TP = np.zeros((np.shape(labeltrain)))
Tr_TN = np.zeros((np.shape(labeltrain)))
Tr_FP = np.zeros((np.shape(labeltrain)))
Tr_FN = np.zeros((np.shape(labeltrain)))
Tr_Acc_total = np.zeros((np.shape(labeltrain)[0]))
Tr_Acc_std = np.zeros((np.shape(labeltrain)))
Tr_TP_std = np.zeros((np.shape(labeltrain)))
Tr_TN_std = np.zeros((np.shape(labeltrain)))
Tr_FP_std = np.zeros((np.shape(labeltrain)))
Tr_FN_std = np.zeros((np.shape(labeltrain)))
Tr_Acc_total_std = np.zeros((np.shape(labeltrain)[0]))
Tr_Acc_cv = np.zeros((np.shape(labeltrain)))
Tr_TP_cv = np.zeros((np.shape(labeltrain)))
Tr_TN_cv = np.zeros((np.shape(labeltrain)))
Tr_FP_cv = np.zeros((np.shape(labeltrain)))
Tr_FN_cv = np.zeros((np.shape(labeltrain)))
Tr_Acc_total_cv = np.zeros((np.shape(labeltrain)[0]))
#for l in np.shape(labeltrain)[0]:
#    for i in range(np.shape(labeltrain)[1]):
#        Tr_Acc[l][i] = mean(sim_result[0:l][i]-labeltrain[0:l][i])
#        Tr_TP[l][i] = sum((sim_result[0:l][i]==1)*(labeltrain[0:l][i]==1))/sum(labeltrain[0:l][i]==1)
for l in range(1,np.shape(labeltrain)[0]):
    for i in range(np.shape(labeltrain)[1]):
        Tr_Acc[l][i] = ones((np.size(sim_result[l,i])))-mean(sim_result[0:l,i]-labeltrain[0:l,i])/labeltrain[l,i] #Tr_Acc[l][i] = mean(sim_result[0:l][i]-labeltrain[0:l][i])
        Tr_TP[l][i] = (sum((sim_result[0:l,i]==1)*(labeltrain[0:l,i]==1))+1)/(sum(labeltrain[0:l,i]==1)+1) 
        Tr_TN[l][i] = (sum((sim_result[0:l,i]==0)*(labeltrain[0:l,i]==1))+1)/(sum(labeltrain[0:l,i]==1)+1) 
        Tr_FP[l][i] = (sum((sim_result[0:l,i]==1)*(labeltrain[0:l,i]==0))+1)/(sum(labeltrain[0:l,i]==0)+1) 
        Tr_FN[l][i] = (sum((sim_result[0:l,i]==0)*(labeltrain[0:l,i]==0))+1)/(sum(labeltrain[0:l,i]==0)+1) 
        Tr_Acc_std[l][i] = std(Tr_Acc[0:l,i])
        Tr_TP_std[l][i] = std(Tr_TP[0:l,i])
        Tr_TN_std[l][i] = std(Tr_TN[0:l,i])
        Tr_FP_std[l][i] = std(Tr_FP[0:l,i])
        Tr_FN_std[l][i] = std(Tr_FN[0:l,i])
        Tr_Acc_cv[l][i] = Tr_Acc_std[l,i]/Tr_Acc[l,i]
        Tr_TP_cv[l][i] = Tr_TP_std[l,i]/Tr_TP[l,i]
        Tr_TN_cv[l][i] = Tr_TN_std[l,i]/Tr_TN[l,i]
        Tr_FP_cv[l][i] = Tr_FP_std[l,i]/Tr_FP[l,i]
        Tr_FN_cv[l][i] = Tr_FN_std[l,i]/Tr_FN[l,i]
    Tr_Acc_total[l] = mean(sim_result[0:l]-labeltrain[0:l]) 
    Tr_Acc_total_std[l] = std(sim_result[0:l]-labeltrain[0:l])
    Tr_Acc_total_cv[l] = Tr_Acc_total_std[l]/Tr_Acc_total[l]

    
#plot
plt.figure()
plt.pcolor(sim_result)
plt.title('TrainingResult')
plt.figure()
plt.pcolor(labeltrain)
plt.title('TrainingLabel')
plt.figure()
plt.pcolor(abs(sim_result-labeltrain))
plt.title('LabelResidule')

plt.figure()
plt.plot(Tr_Acc[::-1])
plt.title('TrainingAccuracy')
#plt.legend()
plt.figure()
plt.plot(Tr_TP)
plt.title('TrainingTruePositives')
plt.figure()
plt.plot(Tr_TN)
plt.title('TrainingTrueNegatives')
plt.figure()
plt.plot(Tr_FP)
plt.title('TrainingFalsePositives')
plt.figure()
plt.plot(Tr_FN)
plt.title('TrainingFalseNegatives')
plt.figure()
plt.plot(Tr_Acc_total)
plt.title('TrainingAccuracyTotal')

for i in range(np.shape(Tr_Acc_cv)[1]):
    plt.figure()
    x = range(np.shape(Tr_Acc_cv)[0]-1)
    y = Tr_Acc_cv[:-1,i]
    yerror = Tr_Acc_std[:-1,i]
    plt.errorbar(x,y,yerror)
    plt.title('TrainingAccuracyCV for feature{}'.format(i))
    plt.figure()
    y = Tr_TP_cv[range(np.shape(Tr_Acc_cv)[0]-1),i]
    yerror = Tr_TP_std[range(np.shape(Tr_Acc_cv)[0]-1),i]
    plt.errorbar(x,y,yerror)
    plt.title('TrainingTruePositivesCV for feature{}'.format(i))
    plt.figure()
    y = Tr_Acc_cv[range(np.shape(Tr_Acc_cv)[0]-1),i]
    yerror = Tr_Acc_std[range(np.shape(Tr_Acc_cv)[0]-1),i]
    plt.plot(x,y,yerror)
    plt.title('TrainingTrueNegativesCV for feature{}'.format(i))
    plt.figure()
    y = Tr_Acc_cv[range(np.shape(Tr_Acc_cv)[0]-1),i]
    yerror = Tr_Acc_std[range(np.shape(Tr_Acc_cv)[0]-1),i]
    plt.plot(x,y,yerror)
    plt.title('TrainingFalsePositivesCV for feature{}'.format(i))
    plt.figure()
    y = Tr_Acc_cv[range(np.shape(Tr_Acc_cv)[0]-1),i]
    yerror = Tr_Acc_std[range(np.shape(Tr_Acc_cv)[0]-1),i]
    plt.plot(x,y,yerror)
    plt.title('TrainingFalseNegativesCV for feature{}'.format(i))
plt.figure()
y = Tr_Acc_total_cv[range(np.shape(Tr_Acc_cv)[0]-1)]/max(Tr_Acc_total_cv)
yerror = Tr_Acc_total_std[range(np.shape(Tr_Acc_cv)[0]-1)]
plt.errorbar(x, y, yerror)
plt.title('TrainingAccuracyTotalCV')


# Test
# Add RANDOMIZED BENCHMARKING in testing            
D = []
#Number of qubits
nQ = 2
#There are 2 qubits: Q0,Q1.
#Number of seeds (random sequences)
nseeds = 5
#Number of Cliffords in the sequence (start, stop, steps)
nCliffs = np.arange(1,100,10)
#2Q RB on Q0,Q2 and 1Q RB on Q1
rb_pattern = [[0,2],[1]]
#Do three times as many 1Q Cliffords
length_multiplier = [1,3]
#Add QFT to compute energy for simulation and tunelling probaability at first state
#Treating the insulating oxide layer as a finite-height potential barrier
E1 = []
beta = []
L1 = []
Prob = []
hbar = 0.1973 #keVnm/c
m = 511 #keV/c^2
U0 = 10 #eV
const = 2*m/hbar**2
for l in testidx:
        for i in range(8): #range(np.shape(W)[0]):         
            if i < np.shape(labeltrain)[1]:
                x = np.array(dataset[l][int(i*width/6):int(i*width/6+width)],dtype = float)
                Wtemp = W[i][:]                
            else:
                x = np.array(dataset[l][int(-width-1):-1],dtype = float)
                Wtemp = W[i][int(-width-1):-1]
            eps = 1/10000
            Dtemp = 0
            qr2 = qiskit.QuantumRegister(2)
#            cr = qiskit.ClassicalRegister(1)   
#            qc = QuantumCircuit(qr, cr)
            qc2 = QuantumCircuit(qr2)   
            qc2.x(0) 
            qc2.x(1)   
            #%config InlineBackend.figure_format = 'svg' # Makes the images fit            
            qc2.h(qr2[0])
            qc2.cu1(pi/2, 0, 1)
            qc2.h(qr2[1])
  #          qc2.swap(0,1)
            backend = Aer.get_backend("statevector_simulator")
            statevector = execute(qc2, backend=backend).result().get_statevector()
            Etempt = np.dot(statevector.conjugate(),statevector)
            betatempt = np.sqrt(const*(U0-Etempt))
            E1.append(Etempt/abs(Etempt))
            beta.append(betatempt)            
#            job = execute(qc2, backend=backend, shots=shots, optimization_level=1)
#            counts = job.result().get_counts()
#            plot_histogram(counts)
            for steps in range(50):
                pred = abs(ones((np.shape(x)))/(1-np.exp(Wtemp[0:len(x)],x)))
                dw = x*(pred - label1[l][i])
                dw *= eta/len(x)
                Wtemp[0:len(x)] = Wtemp[0:len(x)] - dw
                #if np.mean(abs(ones((np.shape(x)))/(1+np.exp(dot(Wtemp,x))) - label1[l][i])) < tol:
                #    if i < np.shape(W)[1]:
                #        W[i][:] = Wtemp
                #    else:
                #        W[i][int(-width-1):-1] = Wtemp                        
                #    break
                #test with Bayes optimization
                # (convergence analysis with KL)
                #p: target distribution, q data distribution
                p = max(max(abs(pred-label1[l][i])),eps)
                q = max(max(abs(ones((np.shape(x)))/(1-np.exp(Wtemp[0:len(x)],x)))-label1[l][i]), eps)
                if KL(pred,q) < Dtemp:
                    if i < np.shape(W)[1]:
                        W[i][0:len(x)] = Wtemp[0:len(x)]
                    else:
                        W[i][0:len(x)] = Wtemp[0:len(x)] 
                    Dtemp = KL(q,p)
                    break
            if np.mean(abs(ones((np.shape(x)))/(1-np.exp(Wtemp[0:len(x)],x))))>0.5:
                Pred = 1
            else:
                Pred = 0
            if np.mean(abs(ones((np.shape(x)))/(1-np.exp(Wtemp[0:len(x)],x)))) <= 1:
                probtempt = np.mean(abs(ones((np.shape(x)))/(1-np.exp(Wtemp[0:len(x)],x))))   
            else:
                probtempt = np.mean(abs(ones((np.shape(x)))/(1-np.exp(Wtemp[0:len(x)],x))))/max(abs(ones((np.shape(x)))/(1-np.exp(Wtemp[0:len(x)],x))))   
            Prob.append(probtempt)
            Ltempt = log10(probtempt/16*U0/Etempt/(1-Etempt/U0)+0.00001)*log10(1)/2/betatempt
            test_result.append(Pred)  
            if abs(Ltempt) <= 1: 
                L1.append(Ltempt)
            elif size(L1) >0:
                L1.append(Ltempt/L1[-1])
            else:
                L1.append(0)
            if Dtemp >= 0:
                D.append(KL(p,q))

plt.figure()
plt.scatter(range(len(E1)),E1)
plt.title('Quantum Dot Energy at status 1')
plt.figure()
plt.scatter(range(len(L1)),L1)
plt.title('Quantum Dot Width')
plt.figure()
plt.scatter(range(len(Prob)),Prob)
plt.title('Quantum Dot Transmition Rate')
                
                
D_result = np.reshape(np.array(D[1:]),(int(np.size(D)/np.shape(W)[0]),np.shape(W)[0]))
plt.figure()
plt.pcolor(D_result)
plt.colorbar()
plt.title('KL time v.s. features')

test_result = np.reshape(np.array(test_result),(np.shape(labeltest)))
                
#accuracy for testing
Te_D_result_mean = D_result
Te_D_result_std = D_result
Te_D_result_cv = D_result
Te_D_result_mean_total = np.mean(D_result,1)
Te_D_result_std_total = np.mean(D_result,1)
Te_D_result_cv_total = np.mean(D_result,1)

Te_Acc = np.zeros((np.shape(labeltest)))
Te_TP = np.zeros((np.shape(labeltest)))
Te_TN = np.zeros((np.shape(labeltest)))
Te_FP = np.zeros((np.shape(labeltest)))
Te_FN = np.zeros((np.shape(labeltest)))
Te_Acc_total = np.zeros((np.shape(labeltest)[0]))
Te_Acc_std = np.zeros((np.shape(labeltest)))
Te_TP_std = np.zeros((np.shape(labeltest)))
Te_TN_std = np.zeros((np.shape(labeltest)))
Te_FP_std = np.zeros((np.shape(labeltest)))
Te_FN_std = np.zeros((np.shape(labeltest)))
Te_Acc_total_std = np.zeros((np.shape(labeltest)[0]))
Te_Acc_cv = np.zeros((np.shape(labeltest)))
Te_TP_cv = np.zeros((np.shape(labeltest)))
Te_TN_cv = np.zeros((np.shape(labeltest)))
Te_FP_cv = np.zeros((np.shape(labeltest)))
Te_FN_cv = np.zeros((np.shape(labeltest)))
Te_Acc_total_cv = np.zeros((np.shape(labeltest)[0]))

for l in range(1,np.shape(D_result)[0]):
    for i in range(np.shape(D_result)[1]):
#        D_result_0[l,i] = D_result[l,i]+np.random.normal(0,1) #D_result_0[l][i]
        qr = qiskit.QuantumRegister(2)
        qc = QuantumCircuit(qr)                  
        qc.h(qr[0])
        qc.u1(qr[0],qr[1])
        qc.h(qr[0])
        test_result
        Te_D_result_mean[l,i] = mean(D_result[0:l,i])
        Te_D_result_std[l,i] = std(D_result[0:l,i])
        if Te_D_result_mean[l,i] == 0:
            Te_D_result_cv[l,i] = (Te_D_result_std[l,i]+1)/(Te_D_result_mean[l,i]+1)     
#        else：
        else:
            Te_D_result_cv[l,i] = Te_D_result_std[l][i]/Te_D_result_mean[l,i]
        Te_Acc[l][i] = ones((np.size(test_result[l,i])))-mean(test_result[0:l,i]-labetest[0:l,i])/labeltest[l,i] #Tr_Acc[l][i] = mean(sim_result[0:l][i]-labeltrain[0:l][i])
        Te_TP[l][i] = (sum((test_result[0:l,i]==1)*(labeltest[0:l,i]==1))+1)/(sum(labeltest[0:l,i]==1)+1) 
        Te_TN[l][i] = (sum((test_result[0:l,i]==0)*(labeltest[0:l,i]==1))+1)/(sum(labeltest[0:l,i]==1)+1)
        Te_FP[l][i] = (sum((test_result[0:l,i]==1)*(labeltest[0:l,i]==0))+1)/(sum(labeltest[0:l,i]==0)+1) 
        Te_FN[l][i] = (sum((test_result[0:l,i]==0)*(labeltest[0:l,i]==0))+1)/(sum(labeltest[0:l,i]==0)+1) 
        Te_Acc_std[l][i] = std(Te_Acc[0:l,i])
        Te_TP_std[l][i] = std(Te_TP[0:l,i])
        Te_TN_std[l][i] = std(Te_TN[0:l,i])
        Te_FP_std[l][i] = std(Te_FP[0:l,i])
        Te_FN_std[l][i] = std(Te_FN[0:l,i])
        Te_Acc_cv[l][i] = Te_Acc_std[l,i]/Te_Acc[l,i]
        Te_TP_cv[l][i] = Te_TP_std[l,i]/Te_TP[l,i]
        Te_TN_cv[l][i] = Te_TN_std[l,i]/Te_TN[l,i]
        Te_FP_cv[l][i] = Te_FP_std[l,i]/Te_FP[l,i]
        Te_FN_cv[l][i] = Te_FN_std[l,i]/Te_FN[l,i]
    Te_D_result_mean_total[l] = mean(Te_D_result_mean[l])
    Te_D_result_std_total[l] = std(Te_D_result_mean[l])

    if Te_D_result_mean_total[l] == 0:
        Te_D_result_cv_total[l] = (Te_D_result_std_total[l]+1)/(Te_D_result_mean_total[l]+1)
    else:
        Te_D_result_cv_total[l] = Te_D_result_std_total[l]/Te_D_result_mean_total[l]

#plot
plt.figure()
plt.plot(Te_D_result_mean)
plt.title('KL convergence mean of testresult')
plt.figure()
plt.plot(Te_D_result_std)
plt.title('KL convergence std of testresult')
for i in range(np.shape(Te_D_result_cv)[1]):
    plt.figure()
    plt.errorbar(range(np.shape(Te_D_result_cv)[0]),y = Te_D_result_cv[:,i],yerr=Te_D_result_std[:,i])
    plt.title('KL coeffitient variance of testresult for feature{}'.format(i))

plt.figure()
plt.plot(Te_D_result_mean_total)
plt.title('KL convergence mean of total testresult vs epoch')
plt.figure()
plt.plot(Te_D_result_std_total)
plt.title('KL convergence std of total testresult vs epoch')
plt.figure()
plt.errorbar(range(np.size(Te_D_result_cv_total)),y = Te_D_result_cv_total,yerr=Te_D_result_std_total)
plt.title('KL coeffitient variance of total testresult vs epoch')

plt.figure()
plt.pcolor(test_result)
plt.title('TestResult')
plt.figure()
plt.pcolor(labeltest)
plt.title('TestLabel')
plt.figure()
plt.pcolor(abs(test_result-labeltest))
plt.title('TestLabelResidule')

plt.figure()
plt.plot(Te_Acc[::-1])
plt.title('TestAccuracy')
#plt.legend()
plt.figure()
plt.plot(Te_TP)
plt.title('TestTruePositives')
plt.figure()
plt.plot(Te_TN)
plt.title('TestTrueNegatives')
plt.figure()
plt.plot(Te_FP)
plt.title('TestFalsePositives')
plt.figure()
plt.plot(Te_FN)
plt.title('TestFalseNegatives')
plt.figure()
plt.plot(Te_Acc_total)
plt.title('TestAccuracyTotal')

for i in range(np.shape(Te_Acc_cv)[1]):
    plt.figure()
    x = range(np.shape(Te_Acc_cv)[0]-1)
    y = Te_Acc_cv[:-1,i]
    yerror = Te_Acc_std[:-1,i]
    plt.errorbar(x,y,yerror)
    plt.title('TestAccuracyCV for feature{}'.format(i))
    plt.figure()
    y = Te_TP_cv[range(np.shape(Te_Acc_cv)[0]-1),i]
    yerror = Te_TP_std[range(np.shape(Te_Acc_cv)[0]-1),i]
    plt.errorbar(x,y,yerror)
    plt.title('TestTruePositivesCV for feature{}'.format(i))
    plt.figure()
    y = Te_Acc_cv[range(np.shape(Te_Acc_cv)[0]-1),i]
    yerror = Te_Acc_std[range(np.shape(Te_Acc_cv)[0]-1),i]
    plt.plot(x,y,yerror)
    plt.title('TestTrueNegativesCV for feature{}'.format(i))
    plt.figure()
    y = Te_Acc_cv[range(np.shape(Te_Acc_cv)[0]-1),i]
    yerror = Te_Acc_std[range(np.shape(Te_Acc_cv)[0]-1),i]
    plt.plot(x,y,yerror)
    plt.title('TestFalsePositivesCV for feature{}'.format(i))
    plt.figure()
    y = Te_Acc_cv[range(np.shape(Te_Acc_cv)[0]-1),i]
    yerror = Te_Acc_std[range(np.shape(Te_Acc_cv)[0]-1),i]
    plt.plot(x,y,yerror)
    plt.title('TestFalseNegativesCV for feature{}'.format(i))
plt.figure()
y = Te_Acc_total_cv[range(np.shape(Te_Acc_cv)[0]-1)]/max(Te_Acc_total_cv)
yerror = Te_Acc_total_std[range(np.shape(Te_Acc_cv)[0]-1)]
plt.errorbar(x, y, yerror)
plt.title('TestAccuracyTotalCV')
