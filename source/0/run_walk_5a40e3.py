# https://github.com/ogorodnikov/m1/blob/181cdb230d39199a72eeef6f57dc5fd9dc38b37d/app/core-service/core/algorithms/runners/run_walk.py
import sys 

sys.path.append('/home/ec2-user/environment/m1/app/core-service/core/algorithms')


from qiskit import Aer
from qiskit import execute

from walk import walk
from walk import walk_reference


# run_values = {'alice_bits': '10101',
#               'alice_bases': 'XXXZX',
#               'eve_bases': 'XZZZX',
#               'bob_bases': 'XXXZZ',
#               'sample_indices': '0, 2'}

run_values = {}
              

circuit = walk(run_values=run_values, task_log=print)

# circuit = walk_reference()


backend = Aer.get_backend('aer_simulator')

job = execute(circuit, backend, shots=1024)

counts = job.result().get_counts()

print(f'WALK circuit:\n{circuit}')
print(f'WALK counts: {counts}')


# run_data = {'Result': {'Counts': counts}, 
#             'Run Values': run_values}

# walk_post_processing(run_data=run_data, task_log=print)