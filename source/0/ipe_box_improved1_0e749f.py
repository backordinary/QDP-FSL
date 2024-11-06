# https://github.com/ugur42/Qiskit-CKS-Algorithm/blob/a45a2c3875766195360ef08ed86c22d35c0d735d/GPE/IPE_Box_Improved1.py

"""
THIS VERSION DOES NOT NEED TO TURN U INTO A MATRIX TO EXPONENT IT,
WE USE THE QISKIT LANGUAGE FOR IT. WE ALSO USE QISKIT TO CREATE cU 
AUTOMATICALLY INSTEWAD BY HAND AS IT IS BADLY IN IPE_BOX

This is the same IPE as in "IPE with HamiltonSiumlation" instead of applying
U 2^k times we apply U^(2k) once the k-th itartation step.

@author: ugsga


What to fix: Use GPE instead of IPE to reduce number of gates and
increase Nr of qubits
"""
from qiskit import *
import numpy as np
from qiskit.visualization import plot_histogram
import qiskit.tools.jupyter
import matplotlib.pyplot as plt
from qiskit.aqua.algorithms import IQPE
from scipy.linalg import expm, sinm, cosm #for the bridge Hamiltonian simulation
from qiskit.extensions import *

from qiskit.quantum_info.operators import Operator

# A = np.array([[1/2+0.j,-1/3],[-1/3,1/2]])
# A =A/np.linalg.norm(A) #Norm the matrix as stated in Problem 1 CKS

def IPE_box(A,m,hist=False, draw=False):
    """
    Iterative Phase Estiamtion for 2x2 Hermitian Matrix
    Input:
        A: hermitian matrix
        m: m- bit desired resbpresentation of output
        hist: return histogram, default == False
        draw: return circuit plot, defualt == False
    Output: 
        Tuple:
            (EW, IPE.png, hist.png)
    """    
    U_list_temp = [0]*m
    U_list = []
    # control = np.array([[1.+0.j,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
    x = np.arange(0,m,1) #save output of classical register here
    U = qiskit.extensions.HamiltonianGate(A, -1,"Hsim") #generate unitary in H-Gate class
                                                        #-1 to get same as expm(A*1j)
    for k in range(0,m):
        U_list_temp[k]=U.power(2**k)
        # control[2:,2:] =  U_list_temp[k]
        # U_list.append(control.copy()) 
    eigenvec = np.linalg.eig(A)
    dim_b = int(np.log2(len(eigenvec[1][1])))

    
    sim = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(1) #build q-register
    b = QuantumRegister(dim_b) #register for |b> 
    c = ClassicalRegister(m) #build classical-register
    circuit = QuantumCircuit(q,b,c) #build curicuit from register "q"
    initial_state = eigenvec[1][1]   # Define initial_state, last index is cahngeable to 1/0
    # print(initial_state)
    circuit.initialize(initial_state, b)
    cU = [0]*m
    for i in range(0,m):
        cU[i] = U_list_temp[i].control() #generate controlled operator U
    for i in range(1,m+1):
        repetitions = 2**(m-i)
        circuit.h([0])
        circuit.append(cU[m-i],list(range(0,dim_b+1))) #apply cU
        if i == 1:
            pass
        else:
            circuit.rz(-2*np.pi*x[i-2]/2**(m-i-1),q[0])
        circuit.h(0)
        circuit.measure(q[0],c[i-1])
        job = execute(circuit, sim, shots=1, memory=True)
        x[m-i] = int(job.result().get_memory()[0][m-i])
        circuit.reset(q[0])
    if draw == True:
        circuit.draw(filename="IPE.png", output="mpl") 
        
    else:
        pass
    n=m
    count0 = execute(circuit, sim,shots=4096).result().get_counts()
    key_new = [str(int(key,2)/2**n) for key in list(count0.keys())]
    count1 = dict(zip(key_new, count0.values()))
    if hist == True:
        fig, ax = plt.subplots(1,2)
        plot_histogram(count0, ax=ax[0])
        plot_histogram(count1, ax=ax[1])
        fig.savefig("hist.png")
        plt.tight_layout()
        plt.close()
        
    else:
        pass
    
    #get maximum of plot
    maxim = count0.most_frequent() #Return the most frequent count
    maxim_dec = int(maxim,2)/2**n # Convert a binary string to a decimal int.
    EW = np.exp(maxim_dec*2*np.pi*1j) #get eigenvalues in mixed for(re and im are mixed)
    EW1 = EW.real
    EW2 = EW.imag
    print("EW1=",EW1, "EW2=",EW2)
    return [EW,"IPE.png","hist.png"]
    
# B = np.array([[1+0.j,2,3,4],[2,1,2,3],[3,2,4,1],[4,3,1,2]])
# B = B/np.linalg.norm(B)
# print(IPE_box(B,6,draw=True,hist=True))


# print(np.matmul(B,B.transpose()))

# print("EW by numpy:",np.linalg.eig(B)[0])
