# https://github.com/jarndejong/FTSWAP_experiment/blob/b9d9d171ea25d712d3f77285119c490b018e46e0/Get_results.py
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 08:59:49 2018

@author: Jarnd
This file checks every job from a previously-ran experiment.
Which experiment is either the last one if 'direct' is set to True,
or is specified by a prompted run_type and circuit_name and date is 'direct' is set to False.
"""

    
#%% Import needed functions from qiskit and import other needed functions
import Functions.results_gathering as rg
from qiskit import register, unregister, get_backend


#%% Register at IBM_Q_Experience
from IBM_Q_Experience.Q_Exp_register import qx_config # Import the token
provider = register(qx_config['APItoken'])

#%% Import the circuit name and run type from the last saved jobdata if direct=True. If direct=False, the circuit name and runtype is prompted
direct = True                                                                   # Set to True for direct importing of the jobids and jobdata from the last saved file without prompting
if direct:
    run_type = rg.store.load_last()['Type']
    circuit_name = rg.store.load_last()['Circuit name']
else:
    run_type = input('Runtype is (enter as string): ')
    circuit_name = input('Circuit name is (enter as string): ')

if run_type == 's':
    backend = get_backend('ibmq_qasm_simulator')
elif run_type == 'r':
    backend = get_backend('ibmqx4')
#%% Load the jobdata and the jobids for the circuit_name and run_type
[jobids, jobdata] = rg.get_jobids_from_file(direct, circuit_name, run_type)
stati = rg.get_status_from_jobids(jobids, printing=True)                        # Get the stati from all different jobs and print them

if 'RUNNING' not in stati:                                                      # If all jobs are done, the results are gathered
    results = rg.get_results_from_jobids(jobids, backend)                       # Gather the results
    calibrations = rg.get_calibration_from_jobids(jobids)                       # Get the calibrations to be saved
    rg.store.save_results(circuit_name, jobdata['Experiment time'],             # Save a dictionary using the save_results method
                          jobdata['Type'], jobdata['Backend'], jobids,
                          jobdata['Tomoset'], jobdata['Batchnumber'],
                          jobdata['Shot number'], results,
                          jobdata['Unitary'], calibrations, notes=None)

#%%
unregister(provider)                                                            # Unregister to prevent conflict