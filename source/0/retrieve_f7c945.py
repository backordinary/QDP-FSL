# https://github.com/ironmanaudi/Machine-Learning-for-QEM/blob/4e645d70418407854f0fb6c91846ca727ea510be/retrieve.py
import qiskit
from qiskit import *

resdata = []
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_16_melbourne')

n = 0
#for ran_job in reversed(backend.jobs(limit=5)):
    #print(n, str(ran_job.job_id()) + " " + str(ran_job.status()))
    #print(n,'aaaaaaaaaaaaaaaaaa', ran_job.result())
    #resdata.append([(ran_job.result())])
    #print(ran_job.result())
    #n = n + 1
#print((list(reversed(backend.jobs(5)))[0]).job_id())
data = ((list(reversed(backend.jobs(5)))[0]).result()).get_counts()
data = list(data)

for i in data:
    print(i)

#print(resdata)

