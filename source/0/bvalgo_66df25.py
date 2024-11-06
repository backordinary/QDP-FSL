# https://github.com/clawyq/qiskit/blob/97735cde61826029d236098342b1503e6352b60f/BValgo.py
'''
Bernstein-Vazirani:
takes as input a string of bits and returns the bitwise product against hidden internal string to return 0 or 1
classical: O(n)

1. init input qubits to |0> state, output qubit to |->.
2. Apply Hadamard gates to input register
3. Query oracle
4. Apply Hadamrd gates to input register
5. Measure
'''
import qiskit
from qiskit import *
from qiskit.tools.monitor import job_monitor

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

hidden_string = '101011'
STR_LENGTH = len(hidden_string)

# setup
# + 1 here for our output qubit, second argument 6 for #classical registers required to store the independent measurements
qc = QuantumCircuit(STR_LENGTH + 1, STR_LENGTH)
# hadamard everything
qc.h(list(range(STR_LENGTH)))

# measurement
qc.x(6)
qc.h(6)

#separates operations of quantum circuit almost like a blocking await
qc.barrier() 

# actual functionality
for idx in range(STR_LENGTH):
  # apply cnot to the 1 positions, setting our output qubit(6) as target
  if str(hidden_string[idx]) == '1':
    qc.cx(idx, 6)

qc.barrier()

# hadamard everything
qc.h(list(range(STR_LENGTH)))

# measurement
qc.measure(list(range(STR_LENGTH)), list(range(STR_LENGTH)))

IBMQ.save_account(os.environ.get("IBM_API_TOKEN"), overwrite=True)
provider = IBMQ.load_account()

device = provider.get_backend('ibmq_16_melbourne')     #we use ibmq_16_melbourne quantum device 
job = execute(qc, backend=device) #we pass our circuit and backend as usual 
job_monitor(job)    #to see our status in queue

counts = job.result().get_counts()
print(f'counts: {counts}')
