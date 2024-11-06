# https://github.com/noahberthusen/benewop-benchmarking/blob/a8881d93636aba285030b90316a6ba79ffa2b5f2/T1_meas_commands.py
from __init__ import *
from qiskit.pulse import pulse_lib
from qiskit.compiler import assemble
from qiskit.pulse.commands.sample_pulse import SamplePulse

import numpy as np
from matplotlib import pyplot as plt

from qiskit import *
from pulse_commands import *

from scipy.optimize import curve_fit

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

def retrieve_T1_time(qb):
    us = 1.0e-6 # Microseconds
    scale_factor = 1e-14

    time_max_us = 450
    time_step_us = 6
    times_us = np.arange(1, time_max_us, time_step_us)

    delay_times_dt = times_us * us / config.dt

    drive_chan = qiskit.pulse.DriveChannel(qb)
    meas_chan = qiskit.pulse.MeasureChannel(qb)
    acq_chan = qiskit.pulse.AcquireChannel(qb)
    measure = measure_and_acquire(qb)

    t1_schedules = []
    for delay in delay_times_dt:
        sched = qiskit.pulse.Schedule(name=f'T1 delay + {delay * config.dt/us} us')
        sched += qiskit.pulse.Play(SamplePulse(Xp(qb).samples, name=f'Xp{qb}'), drive_chan)
        sched |= measure << int(delay)
        t1_schedules.append(sched)

    # Execution settings
    num_shots = 256

    t1_experiment = assemble(t1_schedules,
                            backend=backend, 
                            meas_level=1,
                            meas_return='avg',
                            shots=num_shots,
                            schedule_los=[{drive_chan: defaults.qubit_freq_est[qb]}] * len(t1_schedules))

    job = backend.run(t1_experiment)
    # print(job.job_id())
    t1_results = job.result(timeout=120)

    t1_values = []
    for i in range(len(times_us)):
        t1_values.append(t1_results.get_memory(i)[qb] * scale_factor)
    t1_values = np.real(t1_values)

    fit_params, y_fit = fit_function(times_us, t1_values, 
            lambda x, A, C, T1: (A * np.exp(-x / T1) + C),
            [-3, 3, 100]
            )

    _, _, T1 = fit_params

    return T1

if (__name__ == '__main__'):
    print('Finding relaxation time')
    qb = 0
    time = retrieve_T1_time(qb)
    print(f'Relaxation time of qubit {0} = {time} us')