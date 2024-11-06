# https://github.com/Dheasra/TPIV---EPFL/blob/6d1f3dfa4eb35360b6447ea81c6c067b9f37e3ac/Grover%20TPIVa/grover_2tarOracle_exa.py
import math
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.compiler import transpile 
# from qiskit.compiler.assemble import assemble
# from qiskit.assembler.disassemble import disassemble
# %config InlineBackend.figure_format = 'svg' # Makes the images look nice
import matplotlib.pyplot as plt
import matplotlib

N = 3
Na = 0

slct = 1

Target = 0 #Target bit, 0<=traget<=N-1
Nc = N #nbr of classical bits
ns = 1 #1 = sim w/ noise, 2 = real device, else = sim
n_model = 0 #0 = model from backend, 1= depolarizing error
comp_slct = 0 #Use for sim only! 0 = run on ideal circuit, 1 = run on transpiled circuit 
P = 0.2 #depolarizing probability

def get_noise(p, qubits=1): #from: https://github.com/MIGUEL-LO/Qiskit_depolarazation_channel/blob/master/depo_channel_using_depolarizing_error.ipynb
    
    # This creates the depolarizing error channel,
    # epsilon(P) = (1-P)rho + (P/3)(XrhoX + YrhoY + ZrhoZ).
    depo_err_chan = depolarizing_error((4*p)/3, qubits)

    # Creating the noise model to be used during execution.
    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(depo_err_chan, "measure") # measurement error is applied to measurements

    return noise_model, depo_err_chan

def mcz(crct, n, na):
    ntot = n + na
    #todo: remplacer n, na dans les arguments par Cqubit (liste des indices de qubit de contrôle)
    #todo: pour l'instant le code prend en contrôle tous les autres qubits que le target
    if n > 4:
        #first ccnot gates
        crct.ccx(n-1,n-2, n)
        for i in range(n-2):
            crct.ccx(n-3-i,n+i, n+1+i)
        crct.cz(ntot-1,0) #controlled operation
        #second ccnot gates
        for i in range(n-2):
            crct.ccx(i,ntot-2-i, ntot-1-i)
        crct.ccx(n-1,n-2, n)
    else:
        mcz_alt(crct,n)

def mcz_alt(crct, n):
    if n==3:
        crct.h(0)
        crct.toffoli(1,2,0)
        crct.h(0)
    if n==1:
        crct.z(0)
    if n==2:
        crct.cz(0,1)
    if n==4:
        phi = pi/8
        crct.rz(phi,0)
        #====
        crct.cx(0,1)
        crct.rz(-phi,1)
        crct.cx(0,1)
        crct.rz(phi,1)
        #====
        crct.cx(1,2)
        crct.rz(-phi,2)
        crct.cx(0,2)
        crct.rz(phi,2)
        crct.cx(1,2)
        crct.rz(-phi,2)
        crct.cx(0,2)
        crct.rz(phi,2)
        #====
        crct.cx(2,3)
        crct.rz(-phi,3)
        crct.cx(0,3)
        crct.rz(phi,3)
        crct.cx(1,3)
        crct.rz(-phi,3)
        crct.cx(0,3)
        crct.rz(phi,3)
        crct.cx(2,3)
        crct.rz(-phi,3)
        crct.cx(0,3)
        crct.rz(phi,3)
        crct.cx(1,3)
        crct.rz(-phi,3)
        crct.cx(0,3)
        crct.rz(phi,3)


def Oracle(crct, n, na, slct):
    #Oracle with 2 solutions
    if slct == 0: 
        crct.cz(0,1)
    elif slct == 1: 
        #first single target oracle
        crct.x(0)
        crct.x(1)
        crct.x(2)
        mcz(crct, n, na)
        crct.x(0)
        crct.x(1)
        crct.x(2)
        #second single target oracle
        mcz(crct, n, na)



def GroverIterator(crct, n, na, slct):
    #Oracle
    Oracle(crct, n, na, slct)
    #Hadamard
    for qubit in range(n):
        crct.h(qubit)
    #Conditional phase shift
    for qubit in range(n):
        crct.x(qubit)
    mcz(crct, n, na)
    for qubit in range(n):
        crct.x(qubit)

    #Hadamard
    for qubit in range(n):
        crct.h(qubit)


#Start of the circuit
grvr = QuantumCircuit(N+Na,Nc) #initializing

#Constructing the uniform superposition of states
for qubit in range(N):
    grvr.h(qubit)

grvr.barrier()

#Applying the Grover iterator
for repet in range(math.floor(math.sqrt(N))):
    GroverIterator(grvr, N, Na, slct)
    grvr.barrier()

for n in range(N):
    grvr.measure(n,n)

#visualization of the circuit
# print(grvr) #in the terminal
grvr.draw('mpl')

inst = grvr.data
print(inst)

if comp_slct == 1:
    #Visualizing the compiled circuit 
    provider         = IBMQ.load_account()
    bcknd = provider.get_backend('ibmq_valencia')
    comp_circ = transpile(grvr, bcknd)
    #Printing the circuit
    comp_circ.draw('mpl')
elif comp_slct == 0:
    comp_circ = grvr


#Start of the circuit
grvr = QuantumCircuit(N+Na,Nc) #initializing

#Constructing the uniform superposition of states
for qubit in range(N):
    grvr.h(qubit)

grvr.barrier()

#Applying the Grover iterator
for repet in range(math.floor(math.sqrt(N))):
    GroverIterator(grvr, N, Na, slct)
    grvr.barrier()

for n in range(N):
    grvr.measure(n,n)

#visualization of the circuit
# print(grvr) #in the terminal
grvr.draw('mpl')

inst = grvr.data
print(inst)

if comp_slct == 1:
    #Visualizing the compiled circuit 
    provider         = IBMQ.load_account()
    bcknd = provider.get_backend('ibmq_valencia')
    comp_circ = transpile(grvr, bcknd)
    #Printing the circuit
    comp_circ.draw('mpl')
elif comp_slct == 0:
    comp_circ = grvr

#=== Results ===
#Getting the noise model
shots = 2048 #nbr of runs of the circuit

if ns == 1: #sim with noise
    provider         = IBMQ.load_account()

    backend          = provider.get_backend('ibmq_valencia')
    if n_model == 0:
        noise_model  = NoiseModel.from_backend(backend)
    else:
        noise_model, depo_err_chan = get_noise(P, 1)
    
    simulator        = Aer.get_backend('qasm_simulator')
    # results = execute(grvr, backend=simulator, shots=shots,noise_model = noise_model).result()
    results = execute(comp_circ, backend=simulator, shots=shots,noise_model = noise_model).result()
elif ns == 2: #real device
    provider         = IBMQ.load_account()
    backend         = provider.get_backend('ibmq_valencia')
    job = execute(grvr, backend = backend, shots = 1024, optimization_level= 3)
    job_monitor(job, interval= 2)
    results = job.result()

else: #sim
    backend = Aer.get_backend('qasm_simulator') #selection of the device on which to execute the circuit
    results = execute(grvr, backend = backend, shots = shots).result()


# qobj = assemble(grvr, backend, shots)
# comp_circ = disassemble(qobj)
# comp_circ = comp_circ[0][0]
# comp_circ.draw('mpl')

#Getting the results
answer = results.get_counts()

#ploting
plot_histogram(answer)
plt.show()


#Code snippets

#Iterator for 2 qubits
# for qubit in range(n):
#     crct.z(qubit)
# for s_qubit in range(n-1):
#     for f_qubit in range(n-s_qubit-1):
#         crct.cz(s_qubit,f_qubit+1)
