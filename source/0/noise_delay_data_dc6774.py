# https://github.com/137sc21/137_sc21/blob/557623ccf4e587b50f28b018f44fc17303458978/src/Noise_Delay_Data.py
import csv
import time
from datetime import date, datetime

from qiskit import *

fields = ['Backend', 'Date', 'Jobs in Queue', 'Start Time', 'Finish Time']
with open('../Data/Analysis_Part_One/Noise_Delay_basicCircuit.csv', 'a+') as f:
    write = csv.writer(f)
    write.writerow(fields)
f.close()
machine_list = ['ibmq_belem','ibmq_quito','ibmq_athens','ibmq_5_yorktown','ibmq_santiago']
IBMQ.save_account('XXX',overwrite=True)
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research', group='XXX', project='main')
while True:
    for each in machine_list:

        backend = provider.get_backend(each)
        status = backend.status()
        jobs_in_queue = status.pending_jobs
        circ = QuantumCircuit(5)
        circ.h(0)
        circ.h(3)
        circ.h(4)
        circ.cx(0, 1)
        circ.cx(0, 2)
        circ.cx(3,4)
        today = date.today()
        now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        start = time.time()
        job = execute(circ, backend)
        finish_time = (time.time() - start)
        name = backend.name()
        print(jobs_in_queue)
        print("=========")
        print(start)
        print("=========")
        print(finish_time)

        rows = [name, now, jobs_in_queue, start,finish_time]
        with open('../Data/Analysis_Part_One/Noise_Delay_basicCircuit.csv', 'a+') as f:
            write = csv.writer(f)
            # write.writerow(fields)
            write.writerow(rows)
    time.sleep(1800)