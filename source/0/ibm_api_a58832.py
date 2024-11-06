# https://github.com/CS6620-S21/Asynchronous-quantum-job-submission-framework/blob/308bf505260bbef1e994c6dbb4e8fcf8df203440/poc/ibm_api.py
from async_job.api.object_store import ObjectStore
from qiskit import *
from qiskit import IBMQ
import json
from client import QobjEncoder
from qiskit.compiler import transpile, assemble
from qiskit.qobj.qasm_qobj import QasmQobj as QasmQobj
import os

token=os.getenv('BACKEND_TOKEN')
IBMQ.save_account(token)
IBMQ.load_account() # Load account from disk
print(IBMQ.providers())  # List all available providers

provider = IBMQ.get_provider(hub='ibm-q')
IBMQ.get_provider(group='open')
print("The backends are:")
print(provider.backends())

print("Selecting backend ibmq_qasm_simulator...")
backend = provider.get_backend('ibmq_qasm_simulator')
print(backend)

#Asynchronous using run
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
mapped_circuit = transpile(qc, backend=backend)
qobj = assemble(mapped_circuit, backend=backend, shots=1024)

qasm_obj_dict = qobj.to_dict([True])
qasm_obj_json = json.dumps(qasm_obj_dict, cls=QobjEncoder)
#The above code is to mock the body of the post API request from the client.


load = json.loads(qasm_obj_json)
recieved_qobj = QasmQobj.from_dict(load)

job = backend.run(recieved_qobj)

print("Printing the status of the job")
print(job.status())

backend_temp = job.backend()
backend_temp


print("The job ID of the job is:")
jobID=job.job_id()
print(jobID)

#Getting the job using job ID
print("The job returned by the job ID" + jobID + "is:")
job_returned = backend.retrieve_job(jobID)
print(job_returned)


#Getting the result of the job returned
result = job_returned.result()
print("The result of the job returned is:")
print(result)
result_dict = result.to_dict()
print("**********************************************")
print("result_dict: ", result_dict)
print("Type of result_dict: ", type(result_dict))

result_json = json.dumps(result_dict, indent=4, sort_keys=True, default=str)
print("Type of result json: ", type(result_json))
print(result_json)
# ob = ObjectStore()
# ob.put_object(job_body=result_json, file_name=jobID, bucket_name="completed-bucket-sp21cs6620")
