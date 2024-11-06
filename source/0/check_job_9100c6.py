# https://github.com/takehuge/Qalgorithm/blob/3f26c624ff38e03e915b3949d583cb747d73aca3/QisKit/Check_Job.py
from qiskit import IBMQ

IBMQ.load_account()
print('Available Providers: ')
IBMQ.providers()

