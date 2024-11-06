# https://github.com/jarndejong/FTSWAP_experiment/blob/b9d9d171ea25d712d3f77285119c490b018e46e0/FTSWAP_processtomo.py
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:01:21 2018

@author: Jarnd
"""
print('\033[H\033[J')

"""
This code was inspired/based on the qiskit tutorials provided by IBM, available
at the qiskit-tutorials github. The Q_Exp_register file especially is based on 
the 'process_tomography.py' file.

This code runs tomography experiments on the IBM Q experience.
The circuit for which the tomography is to be run is loaded from circuits.circuit_name, specified at the top of the file.
If the run type is set to 'r' (for real) a real backend is used; if it is set to 's' (for simulation) a simulated backend is used.

There are first made tomography circuits from the loaded circuit. The Pauli basis is used for both preparation and measurement.
These are 18^n different circuits; for more than 1 qubit this is too much to run in one experiment on the IBM Q experience.
All the tomography circuits are divided into n (+ 1 for remainder) batches, n = nr_batches.
Each batch is then sent to the server.
The jobdata is then saved to file specified by the circuit name, the experiment date and the run type.
The jobdata is a list of nr_batches long with as entries a dictionary containing:
    - The jobid to identify the job with the IBM Q servers
    - The experiement date of the specific job
    - The batchnr and the run type
The jobdata is saved together with other info:
    - The circuit name
    - The tomography set
    - The run type
    - The overall experiment date
    - The used backend in the experiment
    - The number of shots per circuit
    - The unitary matrix of the circuit
    - The total number of batches
"""

# importing the needed Qiskit functions
from qiskit import register, execute, get_backend, unregister 
# Import other needed functions
import Functions.Data_storage as store
import Functions.Create_tomo_circuits as tomo
# Import Quantum program of desired circuit
from Circuits.circuit_FTSWAP import Q_program, q, c, Unitary                    # Change the import file for different circuit
circuit_name = Q_program.get_circuit_names()[0]

###############################################################################
# Register at IBM Quantum Experience using token
from IBM_Q_Experience.Q_Exp_register import qx_config
provider = register(qx_config['APItoken'])
###############################################################################
# Simulation or real experimemt? 's' for simulation, 'r' for real
run_type = 's'
notes = ''                                                                      # Optional notes to be stored in the datafile
maximum_credits = 15                                                            # Maximum number of credits PER BATCH
nr_batches = 4                                                                  # Tries that nr of batches, if total number of circuits is not divisible adds one extra batch with the leftovers

###############################################################################
# Set number of shots, timeout, measurement- and preperation basis and backend
shots = 8192                                                                    # shots for every circuit (max is 8192)
# The backend to use in the simulations. Check available_backends() for all backends
backendsim = 'ibmq_qasm_simulator'
# The backed to use for the actual experiments (e.g. the chip)
backendreal = 'ibmqx4'
# Measurement and preparation basis for process tomography
meas_basis, prep_basis = 'Pauli', 'Pauli'

# Set backend based on run_type
if run_type == 's':
    backendname = backendsim
elif run_type == 'r':
    backendname = backendreal
else:
    print('Error, wrong runtype!')

ibmqxbackend = get_backend(backendreal)
job_data = []
################################################################################
# Create tomo set and tomo circuits; put them in the quantum program
[Q_program, tomo_set, tomo_circuits] = tomo.create_tomo_circuits(
    Q_program, circuit_name, q, c, [1, 0], meas_basis, prep_basis)


# Execute all the tomo circuits
batch_size = int(len(tomo_circuits)/nr_batches)
if len(tomo_circuits) % nr_batches != 0:
    nr_batches += 1                                                             # Add an extra batch if the number of tomo circuits is not divisible by the batch number

for i in range(nr_batches):
    run_circuits = tomo_circuits[i*batch_size:(i+1)*batch_size]                 # The names of the circuits to be ran in this batch
    circuit_list = []
    for cir in run_circuits:
        Q_program.get_circuit(cir).name = cir
        circuit_list.append(Q_program.get_circuit(cir))                         # Put the actual circuits in the ordered list            
    print('Batch %d/%d: %s' % (i+1, nr_batches, 'INITIALIZING'))                # The circuits are now ready to be executed
    
    job = execute(circuit_list, backend=backendname, shots=shots, skip_transpiler=True)               # Execute the circuits of this batch (those in circuit_list)
    job_data.append({'Date': job.creation_date,                                 # Put a dictionary of info of the current job in job_data
                     'Jobid': job.id, 'runtype': run_type, 'batchno': i})
    print('Batch %d/%d: %s' % (i+1, nr_batches, 'SENT'))                        # This batch is done and the jobs are saved to job_data and jobs
    

###############################################################################
store.save_jobdata(circuit_name, job_data, tomo_set,                            # Save the data of all jobs (in job_data) to file
                   backendname, shots, nr_batches, run_type, Unitary)
store.save_last(circuit_name, job_data, tomo_set, backendname,                  # Save also to 'last' file
                shots, nr_batches, run_type, Unitary)

unregister(provider)                                                            # Unregister to prevent conflict with other files that register again
