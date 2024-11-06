# https://github.com/raulillo82/TFG-Fisica-2021/blob/401c4bf6e3a7acd9fbe3d9d66553b39dbf58a5f8/grover.py
#!/usr/bin/python3

'''
 * Copyright (C) 2021 Raúl Osuna Sánchez-Infante
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.txt file for details.
''' 
##################
#Needed libraries#
##################

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import qiskit as q
import sys
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
from random import getrandbits

'''
Grover's algorithim. Intro 
'''

#######################
#Functions definitions#
#######################

'''
Usage function
calling the program with "-h" or "--help" will display the help without returning an error (help was intended)
calling the progam with no options or wrong ones, will display the same help but returning an error
Please bear in mind that some combination of options are simply ignored, see the text of this function itself
'''
def usage():
    print("Usage: " + str((sys.argv)[0]) + " i j k l")
    print("i: Number of qubits (2 or 3, will yield error if different)")
    print("j: Number of solutions (only taken into account if i=3, otherwise ignored). Can only be 1 or 2, will yield error otherwise")
    print("k: Number of iterations (only taken into account for i=3 and j=1, othwerise ignored). Can only be 1 or 2, will yield error otherwise")
    print("l: Perform computations in real quantum hardware, can only be 0 (no) or 1 (yes), will yield error otherwise") 
    if len(sys.argv) == 2 and (str((sys.argv)[1]) == "-h" or str((sys.argv)[1]) == "--help"):
        exit(0)
    else:
        exit(1)

'''
Check whether parameter is an integer
'''
def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

'''
Initialization:
Simply apply an H gate to every qubit
'''
def initialize():
    if len(sys.argv) == 1:
        print ("No arguments given")
        usage()
    elif len(sys.argv) > 5 or str((sys.argv)[1]) == "-h" or str((sys.argv)[1]) == "--help" or (not (is_intstring(sys.argv[1]))) or (int((sys.argv)[1]) != 2 and (int((sys.argv)[1]) != 3)):
    #elif (int((sys.argv)[1]) != 2 and (int((sys.argv)[1]) != 3)):
        usage()
    else:
        #print ("Rest of cases")
        for arg in sys.argv[2:]:
            if not is_intstring(arg):
                sys.exit("All arguments must be integers. Exit.")
        qc = q.QuantumCircuit((sys.argv)[1])
        #Apply a H-gate to all qubits in qc
        for i in range(qc.num_qubits):
            qc.h(i)
        qc.barrier()
    return qc

'''
Implement multi controlled Z-gate, easy to reutilize
'''
def mctz(qc):
    qc.h(2)
    qc.mct(list(range(2)), 2)
    qc.h(2)

'''
Oracle metaimplementation
This function will simply call one of the possibles oracles functions
'''
def oracle (qc):
    #Generate some random bits and implement the oracle accordingly with the result
    bits=getrandbits(qc.num_qubits)
    #2 qubits
    if int((sys.argv)[1]) == 2: 
        print("Random bits to search for are (decimal representation): " + str(bits))
        oracle_2_qubits(qc,bits)
    #3 qubits
    elif int((sys.argv)[1]) == 3:
        #Single solution
        if int((sys.argv)[2]) == 1:
            '''
            Explanation:
            less than sqrt(N) iterations will be needed (so will need to "floor" (truncate) the result)
            As 2 < sqrt(8) < 3 --> n=2 for 100% prob. With n=1, p=0.78125=78,125%
            In the classical case, p=1/4=25% (single query followed by a random guess: 1/8 + 7/8 · 1/7 = 1/4 = 25%)
            Classical results with two runs, p=1/8+7/8·1/7+6/8·1/6= 1/4 + 1/8 = 3/8 = 0.375 = 37,5%
            '''
            print("Random bits to search for are (decimal representation): " + str(bits))
            #Check whether 1 or 2 iterations were requested
            if (int((sys.argv)[3]) == 1) or (int((sys.argv)[3]) == 2):
                iterations = int((sys.argv)[3])
                for i in range(iterations):
                    oracle_3_qubits_single_solution(qc,bits)
                    diffusion(grover_circuit)
            #For any other case, wrong arguments were used, exit
            else:
                usage()
        #2 possible solutions
        elif int((sys.argv)[2]) == 2:
            '''
            Explanation:
            less than sqrt(N/M) times (M=2 different results to look for) will be needed (so will need to "floor" (truncate) the result)
            As sqrt(8/2) = 2 --> n=1 for a theoretical 100% prob. In the classical case, 13/28 = 46,4% 
            '''
            #A list instead of a single element will be used, initialize it with the previous value as first element
            bits=[bits]
            #Generate the second element, also randomly
            bits.append(getrandbits(qc.num_qubits))
            #Elements have to be different, regenerate as many times as needed till different
            while bits[0] == bits[1]:
                bits[1]=getrandbits(3)
            #When done, sort the list of random bits. Order does not matter for our upcoming permutations
            bits.sort()
            print("Random bits to search for are (decimal representation): " + str(bits[0]) + " and " + str(bits[1]))
            oracle_3_qubits_2_solutions(qc,bits)
        #Algorithm only implemented for 1 or 2 possible solution(s), exit if something different requested
        else:
            usage()
    #Algorithm only implemented for 1 or 2 qubits, exit if something different requested
    else:
        usage()

'''
Oracle implementation for 2 qubits.
Simply a controlled-Z gate (cz in qiskit).
For qubits different to 1, an x-gate is needed before and after the cz-gate
'''

def oracle_2_qubits(qc,bits):
    if bits == 0: #00
        qc.x(0)
        qc.x(1)
        qc.cz(0, 1)
        qc.x(0)
        qc.x(1)
    elif bits == 1: #01
        qc.x(1)
        qc.cz(0,1)
        qc.x(1)
    elif bits == 2: #10
        qc.x(0)
        qc.cz(0,1)
        qc.x(0)
    elif bits == 3: #11
        qc.cz(0,1)

    qc.barrier()

'''
Oracle implementation for 3 qubits and single solution.
Reference for oracles: https://www.nature.com/articles/s41467-017-01904-7 (table 1)
'''

def oracle_3_qubits_single_solution(qc,bits):
    if bits == 0:
        qc.x(0)
        qc.x(1)
        qc.x(2)
        mctz(qc)
        qc.x(0)
        qc.x(1)
        qc.x(2)
    elif bits == 1:
        qc.x(1)
        qc.x(2)
        mctz(qc)
        qc.x(1)
        qc.x(2)
    elif bits == 2:
        for i in range(0, 3, 2):
            qc.x(i)
        mctz(qc)
        for i in range(0, 3, 2):
            qc.x(i)
    elif bits == 3:
        qc.x(2)
        mctz(qc)
        qc.x(2)
    elif bits == 4:
        qc.x(0)
        qc.x(1)
        mctz(qc)
        qc.x(0)
        qc.x(1)
    elif bits == 5:
        qc.x(1)
        mctz(qc)
        qc.x(1)
    elif bits == 6:
        qc.x(0)
        mctz(qc)
        qc.x(0)
    elif bits == 7:
        mctz(qc)
            
    qc.barrier()

'''
Oracle implementation for 3 qubits and two possible solutions.
Reference for oracles: https://www.nature.com/articles/s41467-017-01904-7 (table 2)
'''

def oracle_3_qubits_2_solutions(qc,bits):
    if (bits[0] == 0 and bits[1] == 1):
        qc.z(1)
        qc.z(2)
        qc.cz(1, 2)
    elif (bits[0] == 0 and bits[1] == 2):
        for i in range(0, 3, 2):
            qc.z(i)
        qc.cz(0, 2)
    elif (bits[0] == 0 and bits[1] == 3):
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.cz(1, 2)
        qc.cz(0, 2)
    elif (bits[0] == 0 and bits[1] == 4):
        qc.z(0)
        qc.z(1)
        qc.cz(0, 1)
    elif (bits[0] == 0 and bits[1] == 5):
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.cz(1, 2)
        qc.cz(0, 1)
    elif (bits[0] == 0 and bits[1] == 6):
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 0 and bits[1] == 7):
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.cz(1, 2)
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 1 and bits[1] == 2):
        qc.z(0)
        qc.z(1)
        qc.cz(1, 2)
        qc.cz(0, 2)
    elif (bits[0] == 1 and bits[1] == 3):
        qc.z(0)
        qc.cz(0, 2)
    elif (bits[0] == 1 and bits[1] == 4):
        for i in range(0, 3, 2):
            qc.z(i)
        qc.cz(1, 2)
        qc.cz(0, 1)
    elif (bits[0] == 1 and bits[1] == 5):
        qc.z(0)
        qc.cz(0, 1)
    elif (bits[0] == 1 and bits[1] == 6):
        qc.z(0)
        qc.cz(1, 2)
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 1 and bits[1] == 7):
        qc.z(0)
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 2 and bits[1] == 3):
        qc.z(1)
        qc.cz(1, 2)
    elif (bits[0] == 2 and bits[1] == 4):
        qc.z(1)
        qc.z(2)
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 2 and bits[1] == 5):
        qc.z(1)
        qc.cz(1, 2)
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 2 and bits[1] == 6):
        qc.z(1)
        qc.cz(0, 1)
    elif (bits[0] == 2 and bits[1] == 7):
        qc.z(1)
        qc.cz(1, 2)
        qc.cz(0, 1)
    elif (bits[0] == 3 and bits[1] == 4):
        qc.z(2)
        qc.cz(1, 2)
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 3 and bits[1] == 5):
        qc.cz(0, 2)
        qc.cz(0, 1)
    elif (bits[0] == 3 and bits[1] == 6):
        qc.cz(1, 2)
        qc.cz(0, 1)
    elif (bits[0] == 3 and bits[1] == 7):
        qc.cz(0, 1)
    elif (bits[0] == 4 and bits[1] == 5):
        qc.z(2)
        qc.cz(1, 2)
    elif (bits[0] == 4 and bits[1] == 6):
        qc.z(2)
        qc.cz(0, 2)
    elif (bits[0] == 4 and bits[1] == 7):
        qc.z(2)
        qc.cz(1, 2)
        qc.cz(0, 2)
    elif (bits[0] == 5 and bits[1] == 6):
        qc.cz(1, 2)
        qc.cz(0, 2)
    elif (bits[0] == 5 and bits[1] == 7):
        qc.cz(0, 2)
    elif (bits[0] == 6 and bits[1] == 7):
        qc.cz(1, 2)
            
    qc.barrier()

'''
Diffusion operator: Flip sign and amplify
For 2 qubits, simply apply H and Z to each qubit, then cz, and then apply H again to each qubit:
'''

def diffusion(qc):
    if qc.num_qubits == 2:
        qc.h(0)
        qc.h(1)
        qc.z(0)
        qc.z(1)
        qc.cz(0,1)
        qc.h(0)
        qc.h(1)
    elif qc.num_qubits == 3:
        qc.h(0)
        qc.x(0)
        qc.h(1)
        qc.x(1)
        qc.h(2)
        qc.x(2)
        # multi-controlled-toffoli
        mctz(qc)
        qc.barrier()
        qc.x(0)
        qc.h(0)
        qc.x(1)
        qc.h(1)
        qc.x(2)
        qc.h(2)

    #qc.barrier()

'''
Add measurements and plot the quantum circuit:
'''
def measure(qc):
    qc.measure_all()
    qc.draw('mpl')
    plt.draw()
    plt.title("Quantum Circuit")

'''
Generate results from quantum simulator (no plotting)
'''
def results_qsim(qc):
    backend = q.Aer.get_backend('qasm_simulator')
    job = q.execute(qc, backend, shots = 1024)
    return job

'''
Generate results from real quantum hardware (no plotting)
'''
def results_qhw(qc):
    '''
    #Only needed if credentials are not stored (e.g., deleted and regeneration is needed
    token='XXXXXXXX' #Use token from ibm quantum portal if needed to enable again, should be stored under ~/.qiskit directory
    q.IBMQ.save_account(token)
    '''
    provider = q.IBMQ.load_account()
    provider = q.IBMQ.get_provider()
    device = q.providers.ibmq.least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                       not x.configuration().simulator and x.status().operational==True))
    print("Running on current least busy device: ", device)

    transpiled_grover_circuit = q.transpile(qc, device, optimization_level=3)
    qobj = q.assemble(transpiled_grover_circuit)
    job = device.run(qobj)
    q.tools.monitor.job_monitor(job, interval=2)

    return job

'''
Plot results
'''
def draw_job (job,title):
    result = job.result()
    counts = result.get_counts()
    plot_histogram(counts)
    plt.draw()
    plt.title(title)

##############################
#End of functions definitions#
##############################

################################
#Program actually starts here!!#
################################

#Initialization
grover_circuit = initialize()
#Generate the oracle randomly according to the command line arguments
oracle(grover_circuit)
#Diffusion
if (not(int(sys.argv[1]) == 3 and int(sys.argv[2]) == 1)):
    diffusion(grover_circuit)
#Add measurements
measure(grover_circuit)
#Generate results in simulator
job_sim = results_qsim(grover_circuit)
#Plot these results
draw_job(job_sim, "Quantum simulator output")
#Generate results in quantum hw if requested
if int(sys.argv[4]) == 1:
    plt.show(block=False)
    plt.draw()
    #Next line needed for keeping computations in background while still seeing the previous plots
    plt.pause(0.001)
    #Generate results in real quantum hardware
    job_qhw = results_qhw(grover_circuit)
    #Plot these results as well
    draw_job(job_qhw, "Quantum hardware output")
#Keep plots active when done till they're closed, used for explanations during presentations
plt.show()
