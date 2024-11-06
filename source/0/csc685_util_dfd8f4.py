# https://github.com/csc-courses/csc685/blob/288cfaf2f66b14334cbe5933156206e34ac53ad7/exercises/csc685_util.py
import qiskit
import time, math

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

qiskit.__qiskit_version__

# your api token from IBM, first time run.
# after that None is good

#api_token = 'abcdefghijklmnopqrstuvwxyz'
api_token = None 

def load_or_save_IBMQ_account(api_token=None):
    global args_g
    if api_token:
        # only needs to be done once
        # then is stored in e.g. ~/.qistkit/qiskitrc
        IBMQ.save_account(api_token)
    provider = IBMQ.load_account()
    return provider

def list_backends(provider):
    global args_g
    backends = provider.backends()
    print('backends available:')
    for be in backends:
        st = be.status()
        if st.operational:
            print(f'\t{be.name()}, pending jobs:{st.pending_jobs}')

            
def run_quantum_circuit_on_backend(quantum_circuit,provider,backend):
    backend = provider.get_backend(backend)
    qobj = assemble(transpile(quantum_circuit, backend=backend), backend=backend)
    job = backend.run(qobj)
    return job


def wait_for_job(backend, job, provider, wait_interval=5):
    backend = provider.get_backend(backend)
    retrieved_job = backend.retrieve_job(job.job_id())
    start_time = time.time()
    job_status = job.status()
    while job_status not in JOB_FINAL_STATES:
        print(f'Status @ {time.time() - start_time:0.0f} s: {job_status.name},'
              f' est. queue position: {job.queue_position()}')
        time.sleep(wait_interval)
        job_status = job.status()

# choose your backend

backend = 'ibmq_qasm_simulator'
#backend = 'ibmq_armonk'
#backend = 'ibmq_vigo'
#backend = 'ibmq_london'

# and so forth ... chose from the results given by provider.backends()


