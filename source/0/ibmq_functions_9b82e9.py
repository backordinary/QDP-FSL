# https://github.com/mar-be/master-thesis-code/blob/79c102a2b685eb1c54ace12cb6df31f32d4a51ae/qiskit_helper_functions/ibmq_functions.py
from qiskit.compiler import transpile, assemble
from qiskit.providers.aer import noise
from qiskit import IBMQ, Aer, execute
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import CouplingMap
import argparse
from qiskit.visualization import plot_gate_map, plot_error_map
from datetime import timedelta, datetime, timezone
import time
import subprocess
import os
import pickle

from qiskit_helper_functions.non_ibmq_functions import read_dict, apply_measurement
from qiskit_helper_functions.conversions import dict_to_array

def load_IBMQ(token,hub,group,project):
    if len(IBMQ.stored_account()) == 0:
        IBMQ.save_account(token)
        IBMQ.load_account()
    elif IBMQ.active_account() == None:
        IBMQ.load_account()
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    return provider

def get_device_info(token,hub,group,project,device_name,fields,datetime):
    dirname = './devices/%s'%datetime.date()
    filename = '%s/%s.pckl'%(dirname,device_name)
    _device_info = read_dict(filename=filename)
    if len(_device_info)==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            subprocess.run(['rm','-r',dirname])
            os.makedirs(dirname)
        provider = load_IBMQ(token=token,hub=hub,group=group,project=project)
        for x in provider.backends():
            if 'qasm' not in str(x):
                device = provider.get_backend(str(x))
                properties = device.properties(datetime=datetime)
                num_qubits = len(properties.qubits)
                print('Download device_info for %d-qubit %s'%(num_qubits,x))
                coupling_map = CouplingMap(device.configuration().coupling_map)
                noise_model = NoiseModel.from_backend(properties)
                basis_gates = noise_model.basis_gates
                _device_info = {'properties':properties,
                'coupling_map':coupling_map,
                'noise_model':noise_model,
                'basis_gates':basis_gates}
                pickle.dump(_device_info, open('%s/%s.pckl'%(dirname,str(x)),'wb'))
            print('-'*50)
        _device_info = read_dict(filename=filename)
    device_info = {}
    for field in fields:
        if field=='device':
            provider = load_IBMQ(token=token,hub=hub,group=group,project=project)
            device = provider.get_backend(device_name)
            device_info[field] = device
        else:
            device_info[field] = _device_info[field]
    return device_info

def check_jobs(token,hub,group,project,cancel_jobs):
    provider = load_IBMQ(token=token,hub=hub,group=group,project=project)

    time_now = datetime.now()
    delta = timedelta(days=0,seconds=0,microseconds=0,milliseconds=0,minutes=0,hours=12,weeks=0)
    time_delta = time_now - delta

    for x in provider.backends():
        if 'qasm' not in str(x):
            device = provider.get_backend(str(x))
            properties = device.properties()
            num_qubits = len(properties.qubits)
            print('%s: %d-qubit, max %d jobs * %d shots'%(x,num_qubits,x.configuration().max_experiments,x.configuration().max_shots))
            jobs_to_cancel = []
            print('QUEUED:')
            print_ctr = 0
            for job in x.jobs(limit=50,status=JobStatus['QUEUED']):
                if print_ctr<5:
                    print(job.creation_date(),job.status(),job.queue_position(),job.job_id(),'ETA:',job.queue_info().estimated_complete_time-time_now)
                jobs_to_cancel.append(job)
                print_ctr+=1
            print('RUNNING:')
            for job in x.jobs(limit=5,status=JobStatus['RUNNING']):
                print(job.creation_date(),job.status(),job.queue_position())
                jobs_to_cancel.append(job)
            print('DONE:')
            for job in x.jobs(limit=5,status=JobStatus['DONE'],start_datetime=time_delta):
                print(job.creation_date(),job.status(),job.error_message(),job.job_id())
            print('ERROR:')
            for job in x.jobs(limit=5,status=JobStatus['ERROR'],start_datetime=time_delta):
                print(job.creation_date(),job.status(),job.error_message(),job.job_id())
            if cancel_jobs:
                for i in range(3):
                    print('Warning!!! Cancelling jobs! %d seconds count down'%(3-i))
                    time.sleep(1)
                for job in jobs_to_cancel:
                    print(job.creation_date(),job.status(),job.queue_position(),job.job_id())
                    job.cancel()
                    print('cancelled')
            print('-'*100)

def noisy_sim_circ(circuit,token,hub,group,project,device_name):
    backend = Aer.get_backend('qasm_simulator')
    qc = apply_measurement(circuit=circuit)
    num_shots = max(1024,2**circuit.num_qubits)
    backend_options = {'max_memory_mb': 2**30*16/1024**2}

    device_info = get_device_info(token=token,hub=hub,group=group,project=project,device_name=device_name,
    fields=['device','basis_gates','coupling_map','properties','noise_model'])

    device = device_info['device']
    basis_gates = device_info['basis_gates']
    coupling_map = device_info['coupling_map']
    properties = device_info['properties']
    noise_model = device_info['noise_model']
    
    mapped_circuit = transpile(qc,backend=device,layout_method='noise_adaptive')

    noisy_qasm_result = execute(experiments=mapped_circuit,
    backend=backend,noise_model=noise_model,
    shots=num_shots,backend_options=backend_options).result()

    noisy_counts = noisy_qasm_result.get_counts(0)
    assert sum(noisy_counts.values())==num_shots
    noisy_counts = dict_to_array(distribution_dict=noisy_counts,force_prob=True)
    return noisy_counts
