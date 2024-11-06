# https://github.com/VoicuTomut/MaxCut_qosf_screening_tasks/blob/373ba650cfcd9f26872211513e597a2b6d08da80/tools.py
#Tools
import  numpy as np 
import itertools

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister 
from qiskit import Aer, execute
from qiskit.quantum_info.operators import Operator

from qiskit.aqua.algorithms import NumPyEigensolver
############################################################################################################################
#Expected value
# Pauli matrix
I=np.array([[1,0],[0,1]],dtype=np.complex128)
X=np.array([[0,1],[1,0]],dtype=np.complex128)
Y=np.array([[0,-1.0j],[1.0j,0]],dtype=np.complex128)
Z=np.array([[1,0],[0,-1]],dtype=np.complex128)

pauli=[I,X,Y,Z]
labels=['I','X','Y','Z']
indice=[0,1,2,3]

# Set signature:
# Z=|0><0|-|1><1|; ZZ=|00><00|-|01><01|-|10><10|+|11><11|   ZI=|00><00|+|01><01|-|10><10|-|11><11| IZ=|00><00|-|01><01|+|10><10|-|11><11|
def proces_counts( counts,z_index):

    z_index.sort(reverse=True) 
    new_counts = {}
    for key in counts:
        new_key = ''
        for index in z_index:
            new_key += key[-1 - index]
        if new_key in new_counts:
            new_counts[new_key] += counts[key]
        else:
            new_counts[new_key] = counts[key]

    return new_counts

# Calculate expectation value from counts 
def expect_z(counts,shots,z_index=[]):
    
    if len(z_index)==0:
        #print("Zeroo")
        return 1
       
    else:
        z_counts=proces_counts(counts,z_index)
    #print(z_counts)
    expectation=0
    for key in z_counts:
        sign=-1
        #print(key)
        if key.count('1')%2==0:
            sign=1
        expectation= expectation+sign*z_counts[key]/shots
    return expectation

# IZXIIYII->IZZIIZII
def measure_qc(qc,Obs):
    m_qc=qc.copy()
    m_qc.barrier()
    for i in range(len(Obs)):
        if(Obs[i]=='Z')or(Obs[i]=='I'):
            m_qc.measure(i,i)
        if(Obs[i]=='X'):
            m_qc.h(i)
            m_qc.measure(i,i)
        if(Obs[i]=='Y'):
            m_qc.rx(np.pi/2,i)
            m_qc.measure(i,i) 
    return m_qc

# Return expected value of an Observable(Obs) for a state prepare by the circuit qc :
def expected(qc,Obs,shots,backend=Aer.get_backend('qasm_simulator')):
    mc=measure_qc(qc,Obs)
    counts=execute(mc,backend=backend,shots=shots).result().get_counts(mc)
    #print("counts:",counts)
    z_index=[]
    for i in range (len(Obs)):
        if(Obs[i]!='I'):
            z_index.append(i)
    #print("Z-index:",z_index)
    return expect_z(counts,shots,z_index)

############################################################################################################################
def value(h,h_label,circ,backend,shots=5000):
    
    val=0
    for i in range(len(h)):
        if h[i]!=0:
            exp=expected(circ,h_label[i],shots=shots,backend=backend)
            val=val+h[i]*exp
            #print('exp for {} ={}'.format(h_label[i],exp))
            
    return (val)


