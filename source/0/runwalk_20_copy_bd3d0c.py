# https://github.com/JaimePSantos/Dissertation-Tex-Code/blob/15544a4334f61e670d1eeee9849fd168c468863d/Coding/Qiskit/ContQuantumWalk/runWalk%20(copy).py
import sys
sys.path.append('../Tools')
from IBMTools import( 
        simul,
        savefig,
        saveMultipleHist,
        printDict,
        plotMultipleQiskit,
        plotMultipleQiskitIbm,
        plotMultipleQiskitIbmSim,
        multResultsSim,
        setProvider,
        leastBusy,
        listBackends,
        getJob)
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from qiskit import( ClassicalRegister,
        QuantumRegister,
        QuantumCircuit,
        execute,
        Aer,
        IBMQ,
        transpile)
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer.noise import NoiseModel
from qiskit.visualization import( plot_histogram,
                        plot_state_city,
                        plot_gate_map, 
                        plot_circuit_layout)
from qiskit.circuit.library import QFT
from math import (log,ceil)
from scipy.fft import fft, ifft
from scipy.linalg import dft, inv, expm, norm
from numpy.linalg import matrix_power
import networkx as nx
plt.rcParams['figure.figsize'] = 11,8
matplotlib.rcParams.update({'font.size' : 15})

def circulant_adjacency(n,v): #--- it computes an adjacency matrix for the circulant graph
    iv = list(range(0,n))
    av = list(range(0,n-1))
    C = np.zeros([n,n])
    for z in range(n):
        C[z,0] = v[iv[z]]
    for x in range(1,n):
        av = iv[0:-1]
        iv[0] = iv[-1]
        iv[1::] = av
        for y in range(0,n):
            C[y,x] = v[iv[y]]
    return C

def unitary_ctqw(gamma, N, A, marked, t): #---
    Oracle = np.zeros([N,N])
    for x in marked:
        Oracle[x,x] = 1
    U = expm(1j*(-gamma*A - Oracle)*t)
    return U

def trotter(gamma, N, A, marked, t, n_trotter):
    O = np.zeros([N,N])
    for x in marked:
        O[x,x] = 1
    U = matrix_power(expm(1j*(-gamma*A)*t/n_trotter)@expm(1j*(- O)*t/n_trotter), n_trotter)
    return U

def init_state(N,initcond): #generalizar isto ?
    psi0 = np.zeros((N,1))
    if initcond == 'sup':
        psi0[int(N/2)-1] = 1/sqrt(2)
        psi0[int(N/2)] = 1/sqrt(2)
    if initcond== '0':
        psi0[int(N/2)] = 1
    return psi0

def init_state(N,initcond): #generalizar isto ?
    psi0 = np.zeros((N,1))
    if initcond == 'sup':
        psi0[int(N/2)-1] = 1/sqrt(2)
        psi0[int(N/2)] = 1/sqrt(2)
    if initcond== '0':
        psi0[int(N/2)] = 1
    return psi0

def final_state(Op,psi0):
    psiN = np.dot(Op,psi0)
    return psiN

def prob_vec(psiN,N):
    probs = np.zeros((N,1))
    for x in range(N):
        probs[x]=psiN[x]*np.conjugate(psiN[x]) #duvida aqui
    return probs

def diagUniOp(N,diagU0):
    qreg = QuantumRegister(N)
    creg = ClassicalRegister(N)
    circ = QuantumCircuit(qreg,name='UniOp')
    circ.diagonal(diagU0,qreg) 
    circ = transpile(circ) 
    return circ

def contCirc(N,diagUniOp):
    qreg = QuantumRegister(3)
    creg = ClassicalRegister(3)
    circ = QuantumCircuit(qreg,creg)
    circ.x(qreg[0])
    circ.append(QFT(N,do_swaps=False,approximation_degree=0,inverse=True), range(N))
    circ.append(diagUniOp,range(N))
    circ.append(QFT(N,do_swaps=False,approximation_degree=0,inverse=False), range(N))
    circ.measure(qreg,creg)
    return circ


#Cycle example
#N = 8
#c = [0,1] + [0 for x in range(N-3)] + [1]
#A = circulant_adjacency(N,c)
#G = nx.from_numpy_matrix(np.array(A))
#nx.draw(G, with_labels = True)
#plt.show()
#print('Degree: ', nx.degree(G),'\nDensity:', nx.density(G)) 

#Cont operator.
N = 8 
NCirc = 3
gamma =  1/(2*np.sqrt(2))
t = 3 
c = [0,1] + [0 for x in range(N-3)] + [1]
qft = dft(N, scale = 'sqrtn')
iqft = inv(qft)

A = iqft@circulant_adjacency(N,c)@qft
diagA = np.diag(A)
U0 = unitary_ctqw(gamma, N, A, [],t)
diagU0 = np.diag(U0).tolist()
U = iqft@U0@qft

backend = Aer.get_backend('qasm_simulator')
shots = 3000
UCirc = diagUniOp(NCirc,diagU0)
continuousCirc = contCirc(NCirc,UCirc)

g = plt.figure(1)
job = execute(continuousCirc,backend=backend,shots=shots)
result = job.result()
counts = result.get_counts()
correctedResult = { str(int(k[::-1],2)) : v/shots for k, v in counts.items()}

plot_histogram(correctedResult)
g.show()


initCond = '0'
initState = init_state(N,initCond)
psiN = final_state(U,initState)
probvec = prob_vec(psiN,N)
f = plt.figure(2)
x = np.linspace(0,7,8)
plt.plot(x,probvec) #plot the lines
plt.title('Caminhada Circulante')
plt.xlabel("Graph Node")
plt.ylabel("Probability")
f.show()

input()


