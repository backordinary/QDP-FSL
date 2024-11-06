# https://github.com/Dheasra/TPIV---EPFL/blob/6d1f3dfa4eb35360b6447ea81c6c067b9f37e3ac/Grover%20TPIVa/grover_noise.py
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
# %config InlineBackend.figure_format = 'svg' # Makes the images look nice
import matplotlib.pyplot as plt
import matplotlib
import statistics
from statistics import mean

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
    if n > 2:
        #first ccnot gates
        crct.ccx(n-1,n-2, n)
        for i in range(n-2):
            crct.ccx(n-3-i,n+i, n+1+i)
        crct.cz(ntot-1,0) #controlled operation
        #second ccnot gates
        for i in range(n-2):
            crct.ccx(i,ntot-2-i, ntot-1-i)
        crct.ccx(n-1,n-2, n)
    if n==1:
        crct.z(0)
    if n==2:
        crct.cz(0,1)

def Oracle(crct, trgt, n, na):
    #Oracle with 2 solutions
    if n == 3:
        if trgt == 0.6: #tragets = |000> and |101>
            for i in range(n):
                crct.z(i)
                crct.cz(0,1)
                crct.cz(1,2)
    #converting target into binary
    str = bin(trgt)
    #removing the header of the string (0b, as it is a binary string)
    str = str[2:]
    #length of the string
    l = len(str)
    #expanding str to length n (n qubits)
    if l < n:
        for k in range(n-l):
            str = '0' + str
    #inverting the string (using slice)
    str = str[::-1]
    #Implemeting the oracle
    for j in range(n-1,-1,-1):
        if str[j] == '0':
            crct.x(j)
    mcz(crct, n, na)
    for j in range(n-1,-1,-1):
        if str[j] == '0':
            crct.x(j)



def GroverIterator(crct, n, na, trgt):
    #Oracle
    Oracle(crct, trgt, n, na)
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

def Grover(crct, N, Na, trgt):
    #Constructing the uniform superposition of states
    for qubit in range(N):
        crct.h(qubit)

    crct.barrier()

    #Applying the Grover iterator
    for repet in range(math.floor(math.sqrt(N))):
        GroverIterator(crct, N, Na, trgt)
        crct.barrier()

    for n in range(N):
        crct.measure(n,n)

# Control panel
Target = 0 #Target bit, 0<=traget<=N-1
Nmax = 12 #max nbr of qubits (>= 2)
ns = 1 #noise: 1 = yes, else = no
n_model = 1 #noise model: 0 = from backend, else = depolarizing channel (1 qubit)
repet = 1

P = 0.2 #Depolarizing probability


tar_rslt = [] #list of probabilities to find the correct target
err_tar = [] #error on above
nse_rslt = [] #list of mean of probabilities to find noise
err_nse = []

provider         = IBMQ.load_account() #required for the noise model
for N in range(2, Nmax+1):
    print(N)
    tar_tmp = []
    nse_tmp = []
    for k in range(repet):
        Na = N-1 #nbr of ancilla qubits
        Nc = N #nbr of classical bits

        #Start of the circuit
        grvr = QuantumCircuit(N+Na,Nc) #initializing

        Grover(grvr, N, Na, Target)


        #=== Results ===
        #Getting the noise model
        shots = 2048 #nbr of runs of the circuit

        if ns == 1:

            backend          = provider.get_backend('ibmq_athens')
            if n_model == 0:
                noise_model  = NoiseModel.from_backend(backend)
            else:
                noise_model, depo_err_chan = get_noise(P, 1)
            simulator        = Aer.get_backend('qasm_simulator')
            results = execute(grvr, backend=simulator, shots=shots,noise_model = noise_model).result()
        else:
            backend = Aer.get_backend('qasm_simulator') #selection of the device on which to execute the circuit
            results = execute(grvr, backend = backend, shots = shots).result()


        answer = results.get_counts()
        an_val = list(answer.values())
        #constructing the list of non targets
        an_val_alt = []
        for i in range(N):
            if i != Target:
                an_val_alt.append(an_val[i])
        
        tar_tmp.append(an_val[Target]/shots)
        nse_tmp.append(max(an_val_alt)/shots)


    tar_rslt.append(mean(tar_tmp))
    nse_rslt.append(mean(nse_tmp))
    err_tar.append((max(tar_tmp) - min(tar_tmp))/repet)
    err_nse.append((max(nse_tmp) - min(nse_tmp))/repet)


N_list = range(2,Nmax+1)

plt.errorbar(N_list, tar_rslt, err_tar, None,'b x', 'black')
plt.errorbar(N_list, nse_rslt, err_nse, None,'r x', 'purple')
plt.grid(True)
plt.legend(['Target', 'Noise (max)'])
plt.xlabel('Number of qubits')
plt.ylabel('Probability')
plt.yscale('log')

plt.show()

# print(answer)
# print(answer['000'])
# print(an_val)
# print(an_val[0])

#ploting
# plot_histogram(answer)
# plt.show()
