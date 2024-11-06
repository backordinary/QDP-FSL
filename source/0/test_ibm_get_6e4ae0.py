# https://github.com/CS6620-S21/Asynchronous-quantum-job-submission-framework/blob/dad0447e100e0619251312e503f919328c839e99/poc/test_ibm_get.py
from qiskit import *
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
# from qiskit import  qiskit.providers.models.BackendStatus

token=os.getenv('BACKEND_TOKEN')
IBMQ.save_account(token)
IBMQ.load_account() # Load account from disk
print(IBMQ.providers())  # List all available providers

provider = IBMQ.get_provider(hub='ibm-q')
IBMQ.get_provider(group='open')
print("The backends are:")
print(provider.backends())

# print("Selecting backend ibmq_qasm_simulator...")
# backend = provider.get_backend('ibmq_qasm_simulator')
print("Selecting backend ibmq_qasm_simulator...")
backend = provider.get_backend('ibmq_qasm_simulator')
print(backend)

#Getting the job using job ID
print("The job returned by the job ID 605e78a02fc7403db7ba812f is:")
job_returned = backend.retrieve_job("605e78a02fc7403db7ba812f")
print(job_returned)

quantum_job_status = job_returned.status()
print(quantum_job_status)
print(type(quantum_job_status))
print(JOB_FINAL_STATES)

#Getting the result of the job returned
if quantum_job_status in JOB_FINAL_STATES :
    result = job_returned.result()
    print("The result of the job returned is:")
    print(result)
else :
    print("Job is not completed..")