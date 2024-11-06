# https://github.com/raulillo82/TFG-Fisica-2021/blob/401c4bf6e3a7acd9fbe3d9d66553b39dbf58a5f8/d-j.py
#!/usr/bin/python3

'''
 * Copyright (C) 2021 Raúl Osuna Sánchez-Infante
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE.txt file for details.
'''

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import qiskit as q
from qiskit.visualization import plot_histogram
from random import getrandbits
import operator
import sys

'''
Deutsch-Josza algorithm solves a problem without a practical aim. However it does show quantum supremacy for SOME problems.
Given a function f(x), it will return either a constant or a balanced result.
Said otherwise, the function received a binary input and will output a binary output. The truth table of the function would be:

Contant cases: a) f(x) yields 0, no matter what the input is. b) f(x) yields 1, no matter what the input is
x f(x)  x f(x)
0  0    0  1
1  0    1  1

Balanced cases: c) f(x) yields x. d) f(x) yields the opposite of x ("not" x, in a classical circuit)
x f(x)  x f(x)
0  0    0  1
1  1    1  0

How many evalutations are needed to find out whether f(x) is balanced or constant?
The answer for the classical case is obvious, 2. It can be demonstrated that for n bits, the answer would be 2^n (all the possible combinations of n bits)
So the complexity would be O(2^n)

In order to evaluate the same problem with a quantum circuit (which is hopefully more efficient), we know all quantum gates need to be reversible (to be explained in the report)
Is f(x) reversible?? Unfortunately not.

This will be bypassed by using an oracle (blackbox). Our oracle will get two inputs (qubits) and yield two outputs (qubits).
Inputs will be q0(x) and q1(y). Outputs will be q1 and (y xor f(x)). If we apply that oracle twice, we get the inputs. So, the oracle is reversible.
XOR is a classical gate, but it does have its equivalent in the quantum world: CNOT gate.

The algorithm uses two of the three special properties of the quantum world: Superposition and Interference (the remaining one would be (Entanglement)
Steps:
    1: Use two qubits, x and y (or q0 and q1). Initial values are always 0 in quantum circuits.
    2: Invert y (or q1) to have a 1 value
    3: Apply H (Hadamard) gates to both qubits, to bring both qubits to a superposition state
    4: Apply the oracle
    5: Apply H gates again to both outputs (interference)
    6: Measure q0. 0 means constant, 1 means balanced

The oracle was applied only once (H gates are "cheap" computationally speaking). Actually, if this expanded to n bits, the Oracle would still only need one evaluation! We would still need 2n qubits to run the algorithm though.
So the complexity of the algorithm in general would be O(1).

Going from O(2^n) to O(1) is an exponential improvement.
'''

'''
Oracle definition. Remember f(x) is the function we want to evaluate, which will be either balanced or constant. There are two possibilites for each of the cases:
    a) Balanced case
    Input   Input   f(x)    (y xor f(x))    Input   Input   f(x)    (y xor f(x))
    x(q0)   y(q1)                           x(q0)   y(q1)              
     0       0       0          0            0       0       1          1
     0       1       0          1            0       1       1          0
     1       0       0          0            1       0       1          1
     1       1       0          1            1       1       1          0
            (y xor f(x))=y                          (y xor f(x))=not (y)

    b) Constant case
    Input   Input   f(x)    (y xor f(x))    Input   Input   f(x)    (y xor f(x))
    x(q0)   y(q1)                           x(q0)   y(q1)              
     0       0       0          0            0       0       1          1
     0       1       0          1            0       1       1          0
     1       0       1          1            1       0       0          0
     1       1       1          0            1       1       0          1
            (y xor f(x))=y XOR x                   (y xor f(x))=y XOR (not(x))
                CNOT of x and y                         CNOT of (not x) and y

As said, we went from a non-reversible function f(x) to our reversible oracle
'''

#Implementation

'''
Usage function
'''

def usage():
    print("Usage: " + str((sys.argv)[0]) + " i ")
    print("i: 0 or 1.")
    print("For only quantum simulator results, use 0. For both simulator and real hardware, use 1. Will yield error for anything else, except -h or --help which will show this help without returning an error")
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
Initialize the circuit
'''
def initialize():
    if len(sys.argv) != 2 or str((sys.argv)[1]) == "-h" or str((sys.argv)[1]) == "--help" or not is_intstring(sys.argv[1]) or (int((sys.argv)[1]) != 0 and (int((sys.argv)[1]) != 1)):
        usage()

    num_qubits = 2
    qc = q.QuantumCircuit(num_qubits,num_qubits) # Step 1
    qc.x(1)    # Step 2
    qc.barrier() # In order to visualize better
    for i in range(num_qubits):
        qc.h(i)    # Step 3
    qc.barrier() # In order to visualize better
    random_oracle(qc)  # Step 4, implement our oracle randomly
    qc.barrier() # In order to visualize better
    for i in range(num_qubits):
        qc.h(i) # Step 5
    qc.barrier() # In order to visualize better
    qc.measure([0,1],[0,1]) # Step 6, add measurements

    #Plot the circuit
    draw_circuit(qc)

    #Return the circuit
    return qc

'''
Plot the quantum circuit
'''
def draw_circuit(qc):
    qc.draw('mpl')
    plt.draw()
    plt.title("Quantum Circuit")

'''
Oracle metaimplementation
This function will simply call one of the possibles oracles functions
Function will receive a bit to (randomly) get one of the each possibilities for constant or balanced oracle
'''

def random_oracle(qc):
    if(getrandbits(1)==0): #This line selects randomly either constant or balanced oracle
        print ("Constant oracle chosen for f(x)")
        constant_oracle(getrandbits(1),qc) #Get one of the constant cases, also randomly
    else:
        print ("Balanced oracle chosen for f(x)")
        balanced_oracle(getrandbits(1),qc) #Get one of the balanced cases, also randomly

'''
Constant oracle function
'''
def constant_oracle(n,qc):
    if (n==0):  # Oracle for the case f(x) = 0. Notice we need nothing in this case, so "pass".
        print ("Constant oracle chosen for f(x)=0")
    else:  # Oracle for the case f(x) = 1. Invert y through the X-gate
        qc.x(1)
        print ("Constant oracle chosen for f(x)=1")

'''
Balanced oracle function
'''
def balanced_oracle(n,qc):
    if (n==0):  # This is the first part of the constant case. Hence a CNOT gate is needed
        qc.cx(0,1)
        print ("Balanced oracle chosen for f(x)=x")
    else: # This is the second part of the constante case. X gate for x and then a CNOT.
        qc.x(0)
        qc.cx(0,1)
        print ("Balanced oracle chosen for f(x)=not(x)")

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

    transpiled_circuit = q.transpile(qc, device, optimization_level=3)
    qobj = q.assemble(transpiled_circuit)
    job = device.run(qobj)
    q.tools.monitor.job_monitor(job, interval=2)

    return job

'''
Plot results
'''
def draw_job (job,title):
    results = job.result()
    counts = results.get_counts()
    plot_histogram(counts)
    plt.draw()
    plt.title(title)
    #print(counts) #This outputs the results and the number of occurrences of each. It should yield only one possible solution for all 1024 cases
    if (len(counts) == 1):
        output=list(counts.keys())[0]
    else:
        output=max(counts.items(), key=operator.itemgetter(1))[0]
        #print (output)
    #print (output)
    if(output=='00' or output=='10'): #Check for x (q0) being 0 for constant
        solution='Oracle (and hence f(x)) is constant'
    else: #Otherwise, balanced
        solution='Oracle (and hence f(x)) is balanced'
    print(solution) #Print the answer to our problem

#Initliaze the quantum circuit for D-J algorithm
dj_circuit = initialize()

#Generate results in simulator
job_sim = results_qsim(dj_circuit)
#Plot these results
draw_job(job_sim, "Quantum simulator output")

if int(sys.argv[1]) == 1:
    plt.show(block=False)
    plt.draw()
    #Next line needed for keeping computations in background while still seeing the previous plots
    plt.pause(0.001)
    #Generate results in real quantum hardware
    job_qhw = results_qhw(dj_circuit)
    #Plot these results as well
    draw_job(job_qhw, "Quantum hardware output")

#Keep plots active when done till they're closed, used for explanations during presentations
plt.show()
