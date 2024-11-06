# https://github.com/ogorodnikov/m1/blob/00589d2a785fe9b36a30e3a1f3e37fe6d21afac8/app/core-service/core/algorithms/runners/run_qae.py
import sys 

sys.path.append('/home/ec2-user/environment/m1/app/core-service/core/algorithms')

from qiskit import Aer
from qiskit import execute

from qae import qae, qae_post_processing


RUN_VALUES = {'bernoulli_probability': '0.3', 
              'precision': '3'}


# Circuit

circuit = qae(run_values=RUN_VALUES, task_log=print)


# Run

backend = Aer.get_backend('aer_simulator')

job = execute(circuit, backend, shots=1024)

counts = job.result().get_counts()


# Post-processing

RUN_DATA = {'Result': {'Counts': counts}, 
            'Run Values': RUN_VALUES}

qae_post_processing(run_data=RUN_DATA, task_log=print)