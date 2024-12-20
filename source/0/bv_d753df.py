# https://github.com/arjunr00/ucla-cs239-s20-qreative/blob/081d68760425fe736d14f01a082badee549eb5fb/HW3/bv.py
print('Loading dependencies .. ', end='', flush=True)
import argparse
import oracle
import os
import matplotlib
import numpy as np
import time
from dotenv import load_dotenv

from qiskit.quantum_info.operators import Operator
from qiskit import QuantumCircuit, execute, Aer, IBMQ, assemble, transpile
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
print('done\n', flush=True)

outfile = open('times/bv.txt', 'a+')

def generate_circuit(n, a, reload, v):
    SAVEDIR = 'plots/'
    CIRCUIT_FILENAME = f'bv_{n}.pdf'

    if (v): print('Creating circuit .. ', end='', flush=True)
    circuit = QuantumCircuit(n+1, n)

    for i in range(n+1):
        circuit.h(i)
    circuit.z(n)

    # Apply barrier 
    circuit.barrier()

    # Apply the inner-product oracle
    a = a[::-1] # reverse s to fit qiskit's qubit ordering
    for q in range(n):
        if a[q] == '0':
            circuit.i(q)
        else:
            circuit.cx(q, n)
            
    # Apply barrier 
    circuit.barrier()

    for i in range(n):
        circuit.h(i)

    # Map the quantum measurement to the classical bits
    circuit.measure(list(range(n)), list(range(n)))
    if (v): print('done\n', flush=True)

    if v:
        print('Circuit:', flush=True)
        print(circuit.draw('text'), flush=True)
        circuit.draw('mpl')
        print(f'Saving circuit to {CIRCUIT_FILENAME} .. ', end='', flush=True)
        if not os.path.exists(SAVEDIR):
            os.mkdir(SAVEDIR)

        matplotlib.pyplot.savefig(f'{SAVEDIR}{CIRCUIT_FILENAME}')
        print('done\n', flush=True)

    return circuit

##################################
###          LOCAL             ###
##################################
def run_circuit(circuit, s):
    if v: print(f'Loading simulator .. ', end='', flush=True)
    simulator = Aer.get_backend('qasm_simulator')
    if v: print('done')
    if v: print(f'Running job .. ', end='', flush=True)
    job = execute(circuit, simulator, shots=s)
    result = job.result()
    if v: print('done')
    return result.get_counts(circuit)

##################################
###           IMBQ             ###
##################################
def load_api_token():                 
    try:
        load_dotenv()
        API_TOKEN = os.getenv('API_TOKEN')
        IBMQ.save_account(API_TOKEN)
        provider = IBMQ.load_account()
        print(f'\nSuccessfully loaded API token for IBMQ\n')
        ibmq_flag = True
    except Exception as e:
        print(e)
        print(f'\nFailed to load API token for IBMQ .. ', end='', flush=True)
        ibmq_flag = False
        print("running local\n")
    return ibmq_flag

def run_on_ibmq(circuit, s):
    print('Loading account .. ', end='', flush=True)
    provider = IBMQ.load_account()
    print('done')
    device = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= n+1 and not x.configuration().simulator and x.status().operational==True))
    print("Running on current least busy device: ", device)

    job = execute(circuit, backend=device, shots=s, optimization_level=3)
    job_monitor(job, interval = 2)

    results = job.result()
    counts = results.get_counts(circuit)
    get_run_time(job)
    print(f'\nTotal counts: {counts}')
    return counts

def check_valid(counts, shots, a, b):
    best = 0
    chosen = None
    for qubits, frequency in counts.items():
        if best < int(frequency):
            best = int(frequency)
            chosen = qubits
    print(f"Qubit chosen: {chosen}")
    return chosen == a

def get_run_time(job):
    data = {}
    tdel = job.time_per_step()['COMPLETED'] - job.time_per_step()['RUNNING']
    print(f'Execution Time .. {tdel}', file=outfile)

parser = argparse.ArgumentParser(description='CS239 - Spring 20 - Deustch-Josza on Qiskit', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.set_defaults(reload=False, balanced=False, verbose=False)
parser.add_argument("--num", "-n", type=int, default=2, help="Set size of input string")
parser.add_argument("--shots", "-s", type=int, default=500, help="Set num of shots")
parser.add_argument("--reload", "-r", action="store_true", help='Reload new U_f matrix')
parser.add_argument("--verbose", "-v", action="store_true", help='Print out measured bits and steps')
args = parser.parse_args()

# Assign local variables to args and intialize BV object
n = args.num
s = args.shots
r = args.reload
v = args.verbose

print("=======================================================", flush=True)
print("         Testing Bernstein-Vazirani Algorithm          ", flush=True)
if v:
    print(f"\n   Running with bit string of size n = {n}", flush=True)
    print(f"                   number of shots s = {s}", flush=True)
    print(f"                   reload U_f matrix = {r}", flush=True)
print("=======================================================\n", flush=True)

# Load IBMQ Account (if fails, then run locally)
ibmq_flag = load_api_token()

a=''
for i in range(n):
    a += '0' if np.random.rand(1,1)[0][0] > 0.5 else '1'
print(a)

print(f'Running with {n} qubits ..', file=outfile)
print(f'a is {a}', file=outfile)
start = time.time()
circuit = generate_circuit(n, a, r, v)
counts = run_on_ibmq(circuit, s) if ibmq_flag else run_circuit(circuit, s)
end = time.time()
print(f'Full execution time .. {end - start}', file=outfile)

# (a, b) = load_ab_lists(n, r)
ret = check_valid(counts, s, a, 0)

print(f'{"success" if ret else "false"}\n', file=outfile)

print(f"Bernstein-Vazirani {'completed successfully!' if ret else 'failed...'}\n", flush=True)
print(f'(Took {end - start:.2f} s to complete.)', flush=True)

# Store times
TIMES_FILE = 'times/bv.csv'

with open(TIMES_FILE, 'r') as file:
    times = file.readlines()

lineno = n
line=times[lineno].split(',')
line[1] = f'{end - start:.4f}\n'
times[lineno] = ','.join(line)

with open(TIMES_FILE, 'w') as file:
    file.writelines(times)
