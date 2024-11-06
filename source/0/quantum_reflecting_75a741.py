# https://github.com/tumichael83/quantum-cognition/blob/70611c67799f0cc6bb8742f4da0e911129548ebc/reflecting%20boundaries/quantum_reflecting.py
# any kind of matrix stuff
from math import ceil,sqrt
from matplotlib import pyplot as plt
import numpy as np
# matrix exponent
from scipy.linalg import expm
#date and time info
import datetime
# Aer is simultor, QC is circuit, execute executes, transpile is to adapt to real machines
from qiskit import Aer, QuantumCircuit, assemble, execute, transpile
# visualization
from qiskit.visualization import *
#
from qiskit.quantum_info.operators import Operator
# to use a backend: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq
from qiskit import IBMQ
#set up backend
IBMQ.load_account()
provider = IBMQ.get_provider(group='yale-uni-1')
config = mybackend.configuration()
numshots = 1000

yale_backends = ['ibmq_armonk',
'ibmq_santiago',
'ibmq_bogota',
'ibmq_lima',
'ibmq_belem',
'ibmq_quito',
'simulator_statevector',
'simulator_mps',
'simulator_extended_stabilizer',
'simulator_stabilizer',
'ibmq_jakarta',
'ibmq_manila',
'ibm_lagos',
'ibm_perth']

def gen_quantum_randwalk(qubits, drift, diffusion, t):
    #create hamiltonian
    h_dimension = 2**qubits

    #drift diagonal
    a = np.zeros((1,h_dimension))
    for i in range(0, h_dimension):
        # placeholder value for drift
        a[0,i] = i*drift

    #diffusion diagonals
    b = np.zeros((1, h_dimension - 1))
    for i in range(0, h_dimension - 1):
        b[0,i] = diffusion

    # add diagonals to matrix
    H = np.diagflat(a) + np.diagflat(b,1) + np.diagflat(b,-1)
    print(H)

    # compute the unitary
    U = expm(-(1j)*H)

    # obtain actual gate
    unitary_operator = Operator(U, input_dims=(h_dimension), output_dims=(h_dimension))

    randwalk = QuantumCircuit (qubits,qubits)

    # lst of qubits
    qlist = []
    for i in range(qubits):
        qlist.append(i)

    # hadamard each qubit
    for i in range(qubits):
        randwalk.h(i)

    # add unitaries
    for i in range(t):
        randwalk.append(unitary_operator, qlist) # add unitary

    randwalk.measure(qlist,qlist)

    return randwalk

# manual assembly like this will let the IBMQ object help us if we get too ambitious
# because it can automatically split up bigger attempts
def run_backend(circuit, mybackend_name):

    if mybackend_name in yale_backends:
        mybackend = provider.get_backend(mybackend_name)
    else:
        mybackend = Aer.get_backend(mybackend_name)

    config = mybackend.configuration()

    print('transpiling for '+config.backend_name+'...')
    trans_c = transpile(circuit,backend=mybackend, basis_gates=config.basis_gates)

    print('assembling for '+config.backend_name+'...')
    qobj = assemble(trans_c, backend=mybackend,shots=numshots)

    print('running on '+config.backend_name+'...')
    job = mybackend.run(qobj)

    return job

def submit_jobs(qubits, drift, diffusion, t, backend):
    # submit all the jobs
    joblist = []
    for i in range(0,t+1):
        randwalk = gen_quantum_randwalk(qubits,drift,diffusion,i)
        job = run_backend(randwalk, backend)

        joblist.append(job)

    return joblist


# graphs the data for a list of completed jobs
def graph_quantum_sim(qubits, drift, diffusion, t, mybackend_name, joblist):

    if mybackend_name in yale_backends:
        mybackend = provider.get_backend(mybackend_name)
    else:
        mybackend = Aer.get_backend(mybackend_name)
    config = mybackend.configuration()

    f = open('reflecting boundaries/'+config.backend_name+'-results.txt', 'w')

    # text file header
    f.write('reflecting boundaries\n'+ config.backend_name + '\nqubits = ' + str(qubits) + '\ndrift = ' + str(drift) + '\ndiffusion = '+str(diffusion))
    f.write('\n')

    #compute num rows required
    nrows = ceil(sqrt(t+1))
    fig, ax = plt.subplots(nrows, nrows, figsize = (16,10))
    fig.suptitle(config.backend_name + " Simulation of Reflecting Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    runtimes = []
    for i, job in enumerate(joblist):

        results = job.result()
        plot_histogram(results.get_counts(),color='midnightblue', ax=ax[i])
        ax[i].title.set_text("QRW timestep " +str(i))
        ax[i].set_ylim([0,1])
        fig.tight_layout()

        f.write('\n-----timestep '+str(i)+'-----\n')
        for n in range(2**qubits):
            k = format(n, '0'+str(qubits)+'b')
            f.write(k+': '+str(results.get_counts()[k])+'\n')


        if mybackend_name in yale_backends:
            runtimes.append(job.time_per_step()['COMPLETED'] - job.time_per_step()['RUNNING'])

    # the distributions
    plt.savefig("./reflecting boundaries/quantum graphs/"+config.backend_name+"-timestep=" + str(t)+'.png', format='png')
    plt.show()

    # the biggest quantum circuit
    randwalk = gen_quantum_randwalk(qubits, drift, diffusion, t)
    transpile(randwalk, basis_gates=config.basis_gates).draw('mpl', filename="./reflecting boundaries/quantum circuit diagrams/"+config.backend_name+"-circuit.png")
    plt.show()

    runtime = sum(runtimes, datetime.timedelta())

    f.write('\ntotal running time: ' + str(runtime))

    f.close()

    return
