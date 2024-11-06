# https://github.com/utzb/ceq/blob/91699cabea3910be2d20c850173a64c1a1607658/src/main.py
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler
from qiskit import QuantumCircuit
import os

service = QiskitRuntimeService(channel="ibm_cloud", instance=os.environ["QISKIT_CRN"], token=os.environ["QISKIT_TOKEN"])

N = 6
qc = QuantumCircuit(N)

qc.x(range(0, N))
qc.h(range(0, N))

for kk in range(N // 2, 0, -1):
    qc.ch(kk, kk - 1)
for kk in range(N // 2, N - 1):
    qc.ch(kk, kk + 1)
qc.measure_all()

print("Circuit to be executed:")
print(qc.draw(output="text"))

print("\nResult:")

program_inputs = {"circuits": [qc], "circuit_indices": [0], "run_options": { "shots": 1024 } }

options = Options(optimization_level=1)

with Session(service=service, backend=""):
    sampler = Sampler(options=options)
    job = sampler.run(circuits=qc, shots=1024)
    print(job.result())

print("\nJobs:")
retrieved_jobs = service.jobs(limit=100)
for rjob in retrieved_jobs:
    print(rjob.job_id, rjob.backend, rjob.creation_date, rjob.status())
    #if (rjob.status() == 'QUEUED'):
    #    rjob.cancel()
    #service.delete_job(rjob.job_id)