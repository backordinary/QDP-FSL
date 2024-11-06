# https://github.com/noahberthusen/benewop-benchmarking/blob/a8881d93636aba285030b90316a6ba79ffa2b5f2/pulse_measurement.py
from __init__ import *
from qiskit.pulse import pulse_lib
from qiskit.compiler import assemble
from qiskit.pulse.commands.sample_pulse import SamplePulse

import numpy as np
from matplotlib import pyplot as plt

from qiskit import *
from pulse_commands import *

# from data_analysis import analyze_data

from scipy.optimize import curve_fit


''' %%%%%%%%%%%%  Pulse Stretching & Measurement program   %%%%%%%%%%%%%%

This program will take an input pulse and run several pulse sequences on the
IBM Q computer while stretching the pulse by different amounts. The program will
then take the result and

Inputs to vary:
qb - qubit to run pulse sequences on
input_pulse - type of pulse to analyze/stretch
num_jobs - number of times to stretch and send input pulse (including no stretch)

Example: If you want to test the X90p gate on qubit 7 and perform a 9th order Richardson Extrapolation
qb = 7
input_pulse = X90p(qb)
num_jobs = 9


'''
qb = 0 # qubit to operate on
input_pulse = Xp(qb) # choose pulse here
num_jobs = 3 # choose number of stretches here
num_gates = 0
plot_title = "Xp default gate, varying gate depth ( Xp*(Xp*Xp)^n )"

drive_samples = 128
drive_sigma = 16
drive_amp = 0.016035
# custom_Xp = lambda i: pulse_lib.gaussian(duration=drive_samples, amp=drive_amp, sigma=drive_sigma, name='Xp')
# gauss_part_Xp = lambda i: SamplePulse(real(Xp(i).samples)) # get only real part of Xp, which is gaussian part


meas_samples = 1200

Xp_samples = Xp(0).samples
print(*Xp_samples, sep='\n') # print out sample values to compare day to day

# begin with no pulses in the experiment
experiments = []

# prepare for measurement by creating measure and acquire pulses, from eugene's code
# meas_pulse = pulse_lib.gaussian_square(duration=meas_samples, amp=0.025,
#                         sigma=4, risefall=25, name='meas_pulse')
# disc = pulse.Discriminator('linear_discriminator',params=pofaults.discriminator.params)
# kern = pulse.Kernel(pofaults.meas_kernel.name)
# acq_cmd=pulse.Acquire(duration=meas_samples, discriminator=disc, kernel=kern)
# meas_and_acq = meas_pulse(device.q[qb].measure)|acq_cmd(device.q,device.mem) # NOTE:  qb was originally 12

meas_and_acq = measure_and_acquire(qb)

'''   %%%%%%%%%%%%% |0> / |1> baseline measurement section %%%%%%%%%%%%%%%%% '''

drive_chan = qiskit.pulse.DriveChannel(qb)
meas_chan = qiskit.pulse.MeasureChannel(qb)
acq_chan = qiskit.pulse.AcquireChannel(qb)

# add (baseline?) measurements of 0/1 state, from eugene's code
schedule = qiskit.pulse.Schedule(name='|0> state')
schedule += meas_and_acq << schedule.duration
experiments.append(schedule) # add zero state to experiments

schedule = qiskit.pulse.Schedule(name='|1> state')
schedule += qiskit.pulse.Play(SamplePulse(Xp(qb).samples, name=f'Xp{qb}'), drive_chan)
schedule += meas_and_acq << schedule.duration
experiments.append(schedule) # add 1 state to experiments


'''  %%%%%%%%%%%%%%%%%%% pulse stretching section %%%%%%%%%%%%%%%%%%%%%%%% '''



# creating pulses to measure

final_time = 0
for i in range(num_jobs):
    # stretch pulse
    # my_pulse = stretch_pulse(input_pulse,dt_added=0,stretched_name = ("stretched input #" + str(i)))
    my_pulse = input_pulse
    # create schedule
    schedule = pulse.Schedule(name=("job #" + str(i)))
    schedule += X90p(qb)(device.q[qb].drive)
    schedule = schedule + (X90m(qb)(device.q[qb].drive) << 10000)

    # add some number of pulses to schedule
    for j in range(i):
        # schedule += my_pulse(device.q[qb].drive) # add some number of pulses to schedule
        schedule += Xm(qb)(device.q[qb].drive)
        schedule += Xp(qb)(device.q[qb].drive)

    # add pulse to list of experiments
    final_time = max(schedule.duration, final_time)
    experiments.append(schedule)

    # draw schedule if desired
    fig = schedule.draw(scaling=20.0,channels_to_plot=[device.q[qb].drive,device.q[qb].measure],label=True)
    plt.show()

experiments = [exp + (meas_and_acq << final_time) for exp in experiments]
'''    %%%%%%%%%%%%%%%%% job assembly and creation %%%%%%%%%%%%%%%%%%%% '''

print(*experiments, sep="\n")

# create Qobj for job
pulse_Qobj = assemble(experiments, pough, meas_level=1, meas_return='avg', shots=1024,rep_time=1000)
job = pough.run(pulse_Qobj)
print("Job Id: " + str(job.job_id()))


''' %%%%%%%%%%%%%%%%%%%% Processing Data Section %%%%%%%%%%%%%%%%%%%%%%%%% '''
#retrieve results
stretch_exp_result = job.result(timeout=3600)
# analyze_data(job.job_id(),qb,plot_title=(plot_title + ", qb = "+str(qb) + ", Job_id="+str(job.job_id())))


''' %%%%%%%%%%%%% data analysis section %%%%%%%%%%%%% '''

results_to_use = stretch_exp_result

num_experiments= len(results_to_use.results) - 2 # minus two for baseline 0/1 measurement

scale_factor=1e-10   # scaling factor for data returned by device

# collect expectation values
result_exp_vals = list([])
for i in range(num_experiments):

    data = results_to_use.get_memory(2+i)[qb]*scale_factor

    my_exp_val = np.mean(data)
    result_exp_vals.append(my_exp_val)

# plot phases
x_points = np.arange(len(result_exp_vals),step=1.0)
plt.plot(np.angle(result_exp_vals,deg=True)) # plot data
plt.plot([0,num_experiments-1],2*[np.mean(np.angle(results_to_use.get_memory(0)[qb]*scale_factor,deg=True))],label="|0> state") # plot 0 state horizontal line
plt.plot([0,num_experiments-1],2*[np.mean(np.angle(results_to_use.get_memory(1)[qb]*scale_factor,deg=True))],label="|1> state") # plot 1 state horizontal line

# fit sine wave to data
sine = lambda x,A,omega,phi,c: A*np.sin(omega*x + phi) + c
# make good initial guess
# amplitude
init_A = np.radians(np.mean(np.angle(results_to_use.get_memory(1)[qb]*scale_factor,deg=True)) - np.mean(np.angle(results_to_use.get_memory(0)[qb]*scale_factor,deg=True))) / 2
init_c = np.radians(np.mean(np.angle(results_to_use.get_memory(1)[qb]*scale_factor,deg=True)) + np.mean(np.angle(results_to_use.get_memory(0)[qb]*scale_factor,deg=True))) / 2

initial_guess = [init_A,0.4,0,init_c]

sine_params, sine_param_covar = curve_fit(sine,x_points,np.angle(result_exp_vals),p0=initial_guess)

# define functions that produced input sample
my_sine = lambda x: sine(x,*sine_params)

# plot fit of sine wave onto data

plt.plot(np.arange(len(result_exp_vals)-0.9,step=0.1), np.degrees(my_sine(np.arange(len(result_exp_vals)-0.9,step=0.1))), color='red', label="sinusoid fit")
freq = sine_params[1]
plt.annotate(("drift freq (radians/step): " + str(freq)),xy=(2, np.degrees(my_sine(2))),textcoords='offset pixels',xytext=(20,-20),fontsize=20)

plt.title(plot_title + ", Job ID: " + job.job_id(), fontsize=20)
plt.legend(loc='upper right', fontsize=20)
plt.xlabel('gate depth (2n+1)', fontsize=20)
plt.ylabel('IQ Phase (degrees)', fontsize=20)
plt.tick_params(labelsize=15)

print("Showing Plot, close plot to continue")
plt.show()