# https://github.com/SPL-LSU/Codes/blob/a28db11b399e6175134e55b973997b67fa44b0df/RoyWIP/Putting_Together.py
#!/usr/bin/env python
# coding: utf-8

# In[2]:


#HOT MESS INCOMING
#Verification
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:34:36 2020

@author: aliza siddiqui

This program creates the diagnostic circuit shown in the following paper:

("Finding Broken Gates in Quantum Circuits---Exploiting Hybrid Machine Learning"
https://arxiv.org/abs/2001.10939)
"""
from qiskit import *
import math as math
import qutip as qt

#Sets U unitary to multi-qubit hadamard
def setMulti_Qubit_Had(qc):
    qc.h(0)
    qc.barrier(0,1,2,3)
    qc.h(1)
    qc.barrier(0,1,2,3)
    qc.h(2)
    qc.barrier(0,1,2,3)
    qc.h(3)
    qc.barrier(0,1,2,3)

#Sets V unitary to Quantum Fourier Transform        
def set_QFT(qc):
    qc.h(3)
    #applies the appropriate rotation gates
    for j in range(3+1,6):
            qc.cu1(math.pi/(2**(j)),j,3)
    qc.h(4)
    #applies the appropriate rotation gates
    for j in range(4+1,6):
            qc.cu1(math.pi/(2**(j)),j,4)
    qc.h(5)
    #applies the appropriate rotation gates
    for j in range(5+1,6):
            qc.cu1(math.pi/(2**(j)),j,5)

    qc.barrier(3,4,5)

#Creating Repeater Circuit        
def createRepeater(qc):
    qc.h(1)
    qc.h(3)
    qc.cx(1,0)
    qc.cx(3,2)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.cx(2,1)
    qc.h(1)
    qc.h(2)
    qc.cx(2,1)
    qc.h(1)
    qc.h(2)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)
    qc.h(2)
    qc.h(0)
    qc.cx(2,0)
    qc.h(2)
    qc.h(0)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)

    qc.barrier(0,1,2,3)
    qc.barrier(4,5,6,7)

#Creates Swap Test portion of circuit    
def swapTest(qc, c):
    qc.h(6) #reducing to 3 quibit since teleportation

    qc.cswap(6, 3, 0)
    qc.cswap(6, 4, 1)
    qc.cswap(6, 5, 2)
    #qc.cswap(8, 7, 3)

    qc.h(6)

    qc.measure(6, c)

#This function teleports a state from qubit one to qubit three
#Returns-nothing
def teleportState(qc, q):
    qc.h(q[1])
    qc.cx(q[1],q[2])
    qc.cx(q[0],q[1])
    qc.h(q[0])
    qc.cx(q[1],q[2])
    qc.cz(q[0],q[2])


def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result    
    

def rabbit(qubits,choice,start):
    basic_0ket=qt.Qobj([[1],[0]])
    basic_1ket=qt.Qobj([[0],[1]])
    if start ==1:
        temp=basic_1ket
    else:
        temp=basic_0ket
    r=1
    while r < qubits:
        if r in choice:
            temp=qt.tensor(temp,basic_1ket)
            temp=tensor_fix(temp)
        else:
            temp=qt.tensor(temp,basic_0ket)
            temp=tensor_fix(temp)
        r+=1
    return temp
    
    
def gen_basis_vectors(n,dims,choice):
    vectors=[]
    basic_states=[]
    bits=int(math.log(n,2))
    #for i in range(n):
        #state=qt.basis(n,i)
        #fock_states.append(state)
    q=rabbit(bits,[],0)
    basic_states.append(q)
    q=rabbit(bits,[1],0)
    basic_states.append(q)
    q=rabbit(bits,[n-1],0)
    basic_states.append(q)
    indexvec=[]
    for i in range(bits):
        indexvec.append(i)
    q=rabbit(bits,indexvec,1)
    basic_states.append(q)
    test_opt1=[basic_states[0],basic_states[1],basic_states[2],basic_states[3]]
    test_opt2=[basic_states[0],basic_states[1],basic_states[2],basic_states[3],basic_states[-1]+basic_states[1]]
    if choice == 1: #Basis states
        vectors = basic_states
        q=qt.Qobj(np.ones(n))
        q=q.unit()
        vectors.append(q)
        vectors.append(state) #there is no two for now
    elif choice ==2:
        vectors=basic_states
    elif choice == 3: #Hadamard option 1
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 4: #QFT option 1
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 5: #Hadamard option 2
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 6: #QFT option 2
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
        vectors.append(state_n)
    return vectors
    
    

def get_ideal(circuit,vectors,qubits,d): 
    #d is the dimension of our vector
    probabilities=[]
    n=2**qubits
    references=[]
    for chi in range(d):
        compare=gen_basis_vectors(n,n,4)
        references.append(compare[chi])
    temparray=[]
    i=0
    for ref in references:
        state=vectors[i]
        final=basic_b(state,circuit)
        prob=dis(final,ref)
        temparray.append(prob)
        i+=1
    probabilities.append(temparray)
    return probabilities


def euclideanDistance(ins1,ins2,length):
    dis=0
    for x in range(length):
        dis += pow((ins1[x]-ins2[x]),2)
    return math.sqrt(dis)


def within_tolerance(tolerance,probvector,idealvector):
    length=4
    scale=1-tolerance
    test=scale*euclideanDistance([0,0,0,0],idealvector,length)
    dist=euclideanDistance(probvector,idealvector,length)
    if dist <= test:
        truth = True
    else:
        truth = False
    return truth

#maybe use Statevector to translate?

#def main():
    #d=4
    #qubits=3
    #n=2**qubits
    #state_creator=[hadamaker(qubits,[1]),qt.cnot(qubits,1,2),qt.cnot(qubits,0,1),hadamaker(qubits,[0]),qt.cnot(qubits,1,2),conv_cz()]
    #state_creator_tags=["Hadamard","CNOT","CNOT2","Hadamard2","CNOT3","Control Z"]
    #circuit=[]

    #for i in range(len(state_creator)):
    #    circuit.append(state_creator[i])
    #    circuit.append(state_creator_tags[i])
    #alt=[]
    #for i in range(len(circuit)):
    #    if type(circuit[i]) == str:
    #        continue
    #    else:
    #        alt.append(gate_troubleshooter(circuit[i],n))
    #vectors=gen_basis_vectors(n,n,2)
    #ideal=get_ideal(alt,vectors,qubits,d)
    #print(ideal[0])
    #tolerance=0.9
    #for i in range(10):
    #    multi=uniform(0,1)
    #    ideal2=ideal[0]
    #    probvector=[]
    #    for x in range(len(ideal2)):
    #        probvector.append(multi*ideal2[x])
    #    truth=within_tolerance(tolerance,probvector,ideal[0])
    #    print(multi)
    #    print(truth)
    #return 0
    
#main()
    
    
    
def main():
    n = 7 #number of qubits in diagnostic circuit
    q = QuantumRegister(n)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    
    setMulti_Qubit_Had(qc)
    #createRepeater(qc)
    teleportState(qc, q)
    set_QFT(qc)
    swapTest(qc, c)
    

   # qc.draw(output = "mpl")
    print(qc)
    
main()


# In[5]:


#Teleportation
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

# Loading your IBM Q account(s)
TOKEN ='4bb7bfab446013a8f953525f3692437a0a314f4f2784431d78ee3eaacd042dfbba453ede40f604c8c996197cde23c9fb509d22c8132c5ac0fd3ffe10c5885bf4'
IBMQ.load_account() # Load account from disk
providers = IBMQ.providers()
provider = IBMQ.get_provider(hub ='ibm-q')
print(provider)
print(provider.backends())
backend = provider.get_backend('ibmq_london')







# In[9]:


from qiskit import *
from qiskit.providers.aer import noise, QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.providers.aer.noise import NoiseModel

#Using simulator from Qiskit Aer
backend_sim = Aer.get_backend('qasm_simulator')



#Using real machine as a backend

#provider = IBMQ.get_provider(hub='ibm-q')
#backend_sim = provider.get_backend('ibmq_london')



#Using real machines as noise model

#provider = IBMQ.get_provider(hub = 'ibm-q') 
#device = provider.get_backend('ibmq_16_melbourne') 
#device = provider.get_backend('ibmqx2') 
#device = provider.get_backend('ibmq_essex')

#properties = device.properties() #will be used to help generate a noise model to run on the simulator
#coupling_map = device.configuration().coupling_map 

number_of_counts = {} #Dictionary for the total number of shots of each state obtained 
number_of_noisy_counts = {} #Dictionary for the total number of shots of each state with noise obtained

#This function returns an image of the circuit that is being run
# Returns-Image of circuit
def showTeleportationCircuit(qc):
    circuit = qc.draw()
    return circuit
    
#This function teleports a state from qubit one to qubit three
#Returns-nothing
def teleportState(qc, q):
    qc.h(q[1])
    qc.cx(q[1],q[2])
    qc.cx(q[0],q[1])
    qc.h(q[0])
    qc.cx(q[1],q[2])
    qc.cz(q[0],q[2])

#This function creates the qubit state one from default state 0
#Returns-nothing
def createOneState(qc, q):
    qc.x(q[0])

#This function creates the qubit plus state from default state 0
#Returns-nothing
def createPlusState(qc, q):
    qc.h(q[0])

#This function creates the qubit minus state from default state 0
#Returns-nothing
def createMinusState(qc, q):
    qc.h(q[0])
    qc.z(q[0])

#This function creates the qubit state plus i from default state 0
def createPlusIState(qc, q):
    qc.h(q[0])
    qc.s(q[0])

#This function creates the qubit state minus i from default state 0
#Returns-nothing
def createMinusIState(qc, q):
    qc.h(q[0])
    qc.z(q[0])
    qc.s(q[0])
    
def createRandomState(qc, q, p):
    qc.u3(p, 0 , 0, q[0])

#This function creates a noise model to apply to a circuit 
# Returns- the amount of counts for each state with noise
def applyNoise(qc):
    gate_lengths = [
    ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
    ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
    ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
    ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
    ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
    ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
    ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
   ]
    noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
    #noise_model = noise.device.basic_device_noise_model(properties, gate_lengths=gate_lengths)
    #print(noise_model)
    
    basis_gates = noise_model.basis_gates
    #simulator = Aer.get_backend('qasm_simulator')
    
    
    result_noise = execute(qc, backend_sim, shots = 10000,
                          noise_model = noise_model,
                          coupling_map = coupling_map,
                          basis_gates = basis_gates).result()
    counts_noise = result_noise.get_counts(qc)
    return counts_noise

#This function teleports the initial qubit state 0 from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing
def teleportZeroState():
    q0 = QuantumRegister(3)
    c0 = ClassicalRegister(1)
    qc0 = QuantumCircuit(q0,c0)
    teleportState(qc0, q0)
    qc0.measure(q0[2],c0)
    job = execute(qc0, backend_sim, shots=10000) #make sure to remove shots when running on actual machine
    counts = job.result().get_counts()
    
    print("ZeroState: ", counts)
    number_of_counts['ZeroState'] = counts
    
    
    #Applying noise
    #noise = applyNoise(qc0)
    #print("NoisyZeroState: ", noise)
    #number_of_noisy_counts['ZeroState']= noise
    
#This function teleports the initial qubit state 1 from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing    
def teleportOneState():
    q1 = QuantumRegister(3)
    c1 = ClassicalRegister(1)
    qc1 = QuantumCircuit(q1,c1)
    createOneState(qc1, q1)
    teleportState(qc1, q1)
    qc1.measure(q1[2],c1)
    job = execute(qc1, backend_sim, shots = 10000)
    counts = job.result().get_counts(qc1)
 
    print("OneState: ", counts)
    number_of_counts['OneState'] = counts
    
    #Applying noise
    #noise = applyNoise(qc1)
    #print("NoisyOneState: ", noise)
    #number_of_noisy_counts['OneState']= noise
    

#This function teleports the initial qubit state |+> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing      
def teleportPlusState():
    qP = QuantumRegister(3)
    cP = ClassicalRegister(1)
    qcP = QuantumCircuit(qP,cP)
    createPlusState(qcP, qP)
    teleportState(qcP, qP)
    qcP.h(qP[2])
    qcP.measure(qP[2],cP)
    job = execute(qcP, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcP)
    
    print("PlusState: ", counts)
    number_of_counts['PlusState']= counts
        
    #Applying noise
    #noise = applyNoise(qcP)
    #print("NoisyPlusState: ", noise)
    #number_of_noisy_counts['PlusState'] = noise

#This function teleports the initial qubit state |-> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing   
def teleportMinusState():
    qM = QuantumRegister(3)
    cM = ClassicalRegister(1)
    qcM = QuantumCircuit(qM,cM)
    createMinusState(qcM, qM)
    teleportState(qcM, qM)
    qcM.h(qM[2])
    qcM.measure(qM[2],cM)
    job = execute(qcM, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcM)
    
    print("MinusState: ", counts)
    number_of_counts['MinusState'] = counts
    
    #Applying noise
    #noise = applyNoise(qcM)
    #print("NoisyMinusState: ", noise)
    #number_of_noisy_counts['MinusState'] = noise

#This function teleports the initial qubit state |+i> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing
def teleportPlusIState():
    qPI = QuantumRegister(3)
    cPI = ClassicalRegister(1)
    qcPI = QuantumCircuit(qPI,cPI)
    createPlusIState(qcPI, qPI)
    teleportState(qcPI, qPI)
    qcPI.sdg(qPI[2])
    qcPI.h(qPI[2])
    qcPI.measure(qPI[2],cPI)
    job = execute(qcPI, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcPI)

    print("PlusIState: ", counts)
    number_of_counts['PlusIState'] = counts
    
    #Applying noise
    #noise = applyNoise(qcPI)
    #print("NoisyPlusIState: ", noise)
    #number_of_noisy_counts['PlusIState'] = noise

#This function teleports the initial qubit state |-i> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing
def teleportMinusIState():
    qMI = QuantumRegister(3)
    cMI = ClassicalRegister(1)
    qcMI = QuantumCircuit(qMI,cMI)
    createMinusIState(qcMI, qMI)
    teleportState(qcMI, qMI)
    qcMI.sdg(qMI[2])
    qcMI.h(qMI[2])
    qcMI.measure(qMI[2],cMI)
    job = execute(qcMI, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcMI)
    
    print("MinusIState: ", counts)
    number_of_counts['MinusIState'] = counts
    
    #Applying noise
    #noise = applyNoise(qcMI)
    #print("NoisyMinusIState: ", noise)
    #number_of_noisy_counts['MinusIState'] = noise
    
    
#This function calculates the fidelity of teleporting an arbitrary state |psi>=sqrt(p)|0>+sqrt(1-p)|1> using:
#p*(fraction of zero outcomes)+(1-p)*(fraction of one outcomes)
#Returns-fidelity of teleporting an arbitrary state
def calculateFidelity(p):
    Values = number_of_noisy_counts['RandomState:|psi>=sqrt(', p, ')|0>+sqrt(1- ', p, ')|1>']
    fidelity = (p* ((Values['0'])/1024)) + ((1-p) * ((Values['1'])/1024))
    return fidelity
    

#This function calculates the average fidelity of teleporting all the pole states: 0, 1, +, -, +i, -i
#Returns-average fidelity
def calculatingFidelity():
    ZeroValues = number_of_counts['ZeroState']
    ZeroProb = (ZeroValues['0'])/10000
    
    OneValues = number_of_counts['OneState']
    OneProb = (OneValues['1'])/10000
    
    PlusValues = number_of_counts['PlusState']
    PlusProbZero = (PlusValues['0'])/10000
    
    MinusValues = number_of_counts['MinusState']
    MinusProbOne = (MinusValues['1'])/10000
    
    PlusIValues = number_of_counts['PlusIState']
    PlusIProbZero = (PlusIValues['0'])/10000
    
    MinusIValues = number_of_counts['MinusIState']
    MinusIProbOne = (MinusIValues['1'])/10000
    
    Fidelity = (ZeroProb + OneProb + PlusProbZero + MinusProbOne + PlusIProbZero + MinusIProbOne)/6
    return Fidelity
    
    
    
    
    
    

def main():  
   print("Trial: ",  (0+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (1+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (2+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (3+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (4+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (5+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (6+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (7+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (8+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (9+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")

main()


# In[ ]:


# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:29:15 2020

GHZ Training Code, based off of Repeater Training Code from 5/20/20

@author: Margarite L. LaBorde
"""
import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
from qutip.qip.algorithms import qft
import time
import csv

#splits the data set into testing and training data. Need to initialize empty vectors first
def handleDataset(array,split,trainingSet=[],testSet=[]):
    #with open(filename,'r') as csvfile:
        #lines = csv.reader(csvfile)
        #dataset=list(lines)
    dataset=array
    for x in range(len(dataset)-1):
        for y in range(len(dataset[0])-1):
            dataset[x][y]=float(dataset[x][y])
            if random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
        #print(trainingSet, 'aaaahhh', testSet)
    return 0

#Finds the euclidean distance bewteen two vectors of length 'length;
def euclideanDistance(ins1,ins2,length):
    dis=0
    for x in range(length):
        dis += pow((ins1[x]-ins2[x]),2)
    return math.sqrt(dis)

#finds the k points nearest a test point
def getKNeighbors(trainingSet,test,k):
    distances=[]
    length=len(test)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(test,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#determines the classes of a vector of neighbors and returns a prediction of a test point's class
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
        sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]
    
#Finds the accuracy of a test set prediction   
def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
        #if testSet[x][-1] != predictions[x]:
            #print("predicted:" + predictions[x],"actual:" + testSet[x][-1])
    return (correct/float(len(testSet)))*100.0
    
def KNN(path,split,k):
    testingset=[]
    trainingset=[]
    
    #Insert csv file name and split here:
    #path=input("Give csv file location, remembering to use forward slashes: ")
    split = split
    split=float(split)
    if split > 1 or split < 0:
        print("Incorrect split input given. Default used.")
        split = 0.66
    handleDataset(path,split,trainingset,testingset)
    
    # generate predictions 
    predictions=[] 
    k=k
    for x in range(len(testingset)): 
        neighbors = getKNeighbors(trainingset,testingset[x],k) 
        result = getResponse(neighbors) 
        predictions.append(result) 
        #print('> predicted=' + repr(result) + ',actual=' + repr(testingset[x][-1]))
        
        
        
    accuracy = getAccuracy(testingset,predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


###############################################################################
#The thing we've been hard-coding everywhere
def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result

id2=qt.Qobj(np.identity(2))

def basic_b(state,array):
    for i in range(len(array)):
        if type(array[i]) is str:
            continue
        else:
            k=qt.Qobj(array[i])
            #print(type(k))
            k=tensor_fix(k)
            #print(k.type,k.shape)
            state=qt.Qobj(state)
            #print(state)
            state=k(state)
    return state
    
def dis(state1,state2):
    (n,m)=state1.shape
    if state1.type is not 'ket' or state2.type is not 'ket':
        print("Error: One or more input to the distance function is not a ket." )
        p_0=0
    else:
        fid=qt.fidelity(state1,state2)
        p_0 = 1/n + (n-1)*fid/n
    return p_0

#messed up hadamard
def multi_qubit_hadamard(regular_hadamard_gate):
    theta = uniform(0.0,math.pi*2.0)
    (n,n) = regular_hadamard_gate.shape
    N = np.int(((np.log(n))/(np.log(2))))
    phase=qt.globalphase(theta,N)
    phase=tensor_fix(phase)
    reg=tensor_fix(regular_hadamard_gate)
    multi_qubit_hadamard = phase*reg
    multi_qubit_hadamard = tensor_fix(multi_qubit_hadamard)    
    return multi_qubit_hadamard
#Looks awful, I promise it isn't. while loops are mostly security
#Just figures out whether there's a hadamard present we can alter
def hadamard_preprocessing(hada):
    storage=hada.full()
    (n,n)=hada.shape
    q=np.log(n)/np.log(2) #number of qubits
    seed=randint(0,q-1)
    forbidden=[] #a vector to hold forbidden seeds
    mag=storage[0][0] #magnitude of the elements in the hadamard
    ongoing=True
    while ongoing:
        i=1
        count=0
        while seed in forbidden: #make sure the seed isn't forbidden
            seed=randint(0,q-1)
            count+=1
            if count == 200: #no infinite loops
                break
        #initialize test unitary
        if seed == 0:
            u1=qt.hadamard_transform(1)
        else:
            u1=id2
        #create test unitary
        while u1.shape != (n,n):
            if i ==seed: #set a hadamard on specified qubit
                u1=qt.tensor(u1,qt.hadamard_transform(1))
                u1=tensor_fix(u1)
            else:
                u1=qt.tensor(u1,id2)
                u1=tensor_fix(u1)
            i+=1
        check=u1*hada
        if check.full()[0][0] > mag: #if there's a hadamard on that qubit, true
            ongoing = False
        elif count == 200:
            print("oops")
            break
        else: #no hadamard on that seed qubit
            forbidden.append(seed)
    return hada,seed

#can feed as input the preprocessing step
def alter_hadamard(hada,seed):
    (n,n)=hada.shape
    theta = uniform(0.0,math.pi*2.0)
    
    #pick a rotation any rotation
    phaser=randint(0,3)
    if phaser ==0:
        gate=qt.phasegate(theta)
    elif phaser == 1:
        gate=qt.rz(theta)
    elif phaser==2:
        gate = qt.ry(theta)
    else:
        gate=qt.globalphase(theta)
        
    #alter gate
    if seed == 0:
        u1=gate
    else:
        u1=id2
    i=1
    while u1.shape != (n,n):
        if i ==seed: #set a alteration on specified qubit
            u1=qt.tensor(u1,gate)
            u1=tensor_fix(u1)
        else:
            u1=qt.tensor(u1,id2)
            u1=tensor_fix(u1)
        i+=1
    final_gate=u1*hada
    return final_gate

#code which give an original unitary gate
def random_unitary_gate(delta,alpha,theta,beta,value):
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
    if value == True:
        gate = gate *qt.Qobj([[0,1],[1,0]])
    else:
        gate = gate
    return gate

def random_angles():
    #gets a random value for each variable in the gate
    choice = randint(1,4)
    unitary_gate = ()
    if choice == 1: #Pauli-Y Gate
        unitary_gate = (0.0,math.pi/2,2*math.pi,math.pi/2,True)
    elif choice == 2: #Pauli-Z Gate
        unitary_gate = (0.0,0.0,math.pi,0.0,True)
    elif choice == 3: #S Gate
        unitary_gate = (-math.pi/2,math.pi,math.pi,math.pi,False)
    elif choice == 4: #T Gate
        unitary_gate = (-math.pi/4,math.pi/2,2*math.pi,0.0,False)
    delta,alpha,theta,beta,value = unitary_gate
    return delta,alpha,theta,beta,value

#code which takes an angle and alters the gate
def random_altered_unitary_gate(delta,alpha,theta,beta,value):
    if delta == 0.0 and alpha == 0.0 and theta == math.pi and value == True:
        angles = ['delta','alpha','beta']
    else:
        angles = ['delta','alpha','theta','beta']
    altered_variable = choice(angles)
    if altered_variable == 'delta':
        delta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'alpha':
        alpha = uniform(0.0,2.0*math.pi)
    if altered_variable == 'theta':
        theta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'beta':
        beta = uniform(0.0,2.0*math.pi)
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
    if value == True:
        gate = gate *qt.Qobj([[0,1],[1,0]])
    else:
        gate = gate
    return gate
   
#gives both an original and altered unitary gate
#can be commented to return oritinal gate, corruspondig altered gate(onle one thing different from original), or both
def unitary_gate(choice):
    delta,alpha,theta,beta,value = random_angles()
    original = random_unitary_gate(delta,alpha,theta,beta,value)
    matrix=original
    if choice:
        altered = random_altered_unitary_gate(delta,alpha,theta,beta,value)
        matrix=altered
    return (matrix,[delta,alpha,theta,beta,value])

def rot(qubits,choice):
    a=randint(0,qubits-2)
    b=randint(a+1,qubits-1)
    k=qt.cnot(qubits,a,b)
    k=k.full()
    if choice:
        k=np.random.permutation(k)
        k=qt.Qobj(k)
        #k=k*k.dag()
        k=tensor_fix(k)
    cn_final=qt.Qobj(k)
    return cn_final

def arb_circuit_generator(length,qubits):
    circuit=[]
    angles=[]
    controls=[]
    id2=np.identity(2)
    id2=qt.Qobj(id2)
    n=2**qubits
    Had = 0
    CNOT = 0
    Ran = 0
    while len(circuit)<2*length:
        seed=randint(1,3)
        if seed ==1:
            temp=qt.hadamard_transform()
            temp=tensor_fix(temp)
            circuit.append(temp)
            circuit.append("Hadamard")
            Had = Had +1
        if seed == 2:
            temp=rot(qubits,False)
            circuit.append(temp)
            circuit.append("CNOT")
            CNOT = CNOT +1
        if seed == 3:
            (temp,ang)=unitary_gate(False) 
            circuit.append(temp)
            circuit.append("Random Unitary")
            angles.append(ang)
            Ran = Ran +1
    for i in circuit:
        if type(i) == str:
            continue
        else:
            i=tensor_fix(i)
    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        elif circuit[i].full().shape == (n,n):
            continue
        else:
            place=randint(1,qubits)
            before=qubits-place
            after=qubits-before-1
            temp=circuit[i]
            temp=qt.Qobj(temp)
            if place == 1:
                circuit[i]=qt.tensor(temp,id2)
            else:
                while before > 0:
                    if circuit[i].full().shape == (n,n):
                        break
                    temp=qt.tensor(id2,temp)
                    temp=tensor_fix(temp)
                    circuit[i]=temp
                    before=before-1
                while after > 0:
                    if circuit[i].full().shape == (n,n):
                        break
                    temp=qt.tensor(temp,id2)
                    temp=tensor_fix(temp)
                    circuit[i]=temp
                    after=after-1
    composition = ["Hadamards:",Had,"CNOT:",CNOT,"Random Unitary:",Ran]
    return (circuit,angles,controls,composition)

def categorize(circuit):
    cat=[]
    it_not=1
    it_h=1
    it_rand=1
    it_id=0
    composition=[]
    
    
    for i in range(len(circuit)):
        if i % 2 != 0:
            continue
        if circuit[i+1] in cat:
            if "Hadamard" in circuit[i+1]:
                it_h+=1
                t=str(it_h)
                t="Hadamard" + t
                cat.append(t)
            elif "CNOT" in circuit[i+1]:
                it_not+=1
                t=str(it_not)
                t="CNOT" + t
                cat.append(t)
            elif "Random Unitary" in circuit[i+1]:
                it_rand+=1
                t=str(it_rand)
                t="Random Unitary" + t
                cat.append(t)
        elif "Identity" in circuit[i+1]:
            it_id +=1
            t=str(it_id)
            t="Measurement" + t
            cat.append(t)
        else:
            cat.append(circuit[i+1])
        i+=2
    return(cat)
    
def rabbit(qubits,choice,start):
    basic_0ket=qt.Qobj([[1],[0]])
    basic_1ket=qt.Qobj([[0],[1]])
    if start ==1:
        temp=basic_1ket
    else:
        temp=basic_0ket
    r=1
    while r < qubits:
        if r in choice:
            temp=qt.tensor(temp,basic_1ket)
            temp=tensor_fix(temp)
        else:
            temp=qt.tensor(temp,basic_0ket)
            temp=tensor_fix(temp)
        r+=1
    return temp
    
def gen_basis_vectors(n,dims,choice):
    vectors=[]
    basic_states=[]
    bits=int(math.log(n,2))
    #for i in range(n):
        #state=qt.basis(n,i)
        #fock_states.append(state)
    q=rabbit(bits,[],0)
    basic_states.append(q)
    q=rabbit(bits,[1],0)
    basic_states.append(q)
    q=rabbit(bits,[n-1],0)
    basic_states.append(q)
    indexvec=[]
    for i in range(bits):
        indexvec.append(i)
    q=rabbit(bits,indexvec,1)
    basic_states.append(q)
    test_opt1=[basic_states[0],basic_states[1],basic_states[2],basic_states[3]]
    test_opt2=[basic_states[0],basic_states[1],basic_states[2],basic_states[3],basic_states[-1]+basic_states[1]]
    if choice == 1: #Basis states
        vectors = basic_states
        q=qt.Qobj(np.ones(n))
        q=q.unit()
        vectors.append(q)
        vectors.append(state) #there is no two for now
    elif choice ==2:
        vectors=basic_states
    elif choice == 3: #Hadamard option 1
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 4: #QFT option 1
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 5: #Hadamard option 2
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 6: #QFT option 2
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
        vectors.append(state_n)
    return vectors

################################################################################

def gate_troubleshooter(gate,n):
    if gate.shape != (n,n):
        gate=tensor_fix(gate)
        while gate.shape !=(n,n):
            seed=randint(0,1)
            if seed ==1:
                gate=qt.tensor(id2,gate)
                gate=tensor_fix(gate)
            else:
                gate=qt.tensor(gate,id2)
                gate=tensor_fix(gate)
        if gate.shape != (n,n):
            print("FUUUUUUUUUUUUUUUUU")
    return gate
        
def h_reassign(hada):
    seed = randint(0,1)
    if seed == 0: #alter whole hadamard
        alt_had=multi_qubit_hadamard(hada)
    if seed == 1: #alter specific hadamard
        (h,seed)=hadamard_preprocessing(hada)
        alt_had=alter_hadamard(hada,seed)
    return alt_had

#Takes as input a circuit w/no str, state vectors,categories a population, number of qubits, and d dimensions of KNN
def colin_mochrie(circuit,angles,vectors,pop,cat,qubits,d,path): 
    probabilities=[]
    n=2**qubits
    index=0
    for j in range(pop):
        references=[]
        for chi in range(d):
            compare=gen_basis_vectors(n,n,4)
            references.append(compare[chi])
        #state=qt.fock(16,0)
        for i in range(len(circuit)):
            gate_holder=circuit[i]
            name=cat[i]
            if "Hadamard" in name:
                alt_gate=h_reassign(gate_holder)
                #print(name,alt_gate)
                alt_gate=gate_troubleshooter(alt_gate,n)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(h_reassign(circuit[i]),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break
            if "Random" in name:
                (delta,alpha,theta,beta,value)=angles[index]
                alt_gate=random_altered_unitary_gate(delta,alpha,theta,beta,value)
                #print(name,alt_gate)
                alt_gate=gate_troubleshooter(alt_gate,n)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(random_altered_unitary_gate(delta,alpha,theta,beta,value),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break
            if "CNOT" in name:
                alt_gate=rot(qubits,True)
                alt_gate=gate_troubleshooter(alt_gate,n)
                #print(name,alt_gate)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(rot(qubits,True),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break
            #final=basic_b(state,circuit)
            temparray=[]
            t=0
            for ref in references:
                state=vectors[t]
                final=basic_b(state,circuit)
                prob=dis(final,ref)
                temparray.append(prob)
                t+=1
            temparray.append(name)
            probabilities.append(temparray)
            with open(path,'a',newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(temparray)
            csvFile.close
            count=0
            while circuit[i] != gate_holder:
                circuit[i]=gate_holder
                count+=1
                if count == 10:
                    break

    return probabilities
    
def categorize_broad(circuit):
    cat=[]
    for i in range(len(circuit)):
        if i % 2 != 0:
            continue
        if circuit[i+1] in cat:
            if "create" in circuit[i+1]:
                cat.append("create")
            elif "Entanglement" in circuit[i+1]:
                cat.append("Entanglement")
            elif "Purification" in circuit[i+1]:
                cat.append("Purification")
        else:
            cat.append(circuit[i+1])
        i+=2
    return(cat)

################################################################################


#affect is list of affected qubits, 0 indexed
#makes an n qubit hadamard, affecting a subset of qubits
def hadamaker(qubits,affected):
    array=[]
    i=0
    while len(array) < qubits:
        if i in affected:
            gate=qt.hadamard_transform()
            gate=tensor_fix(gate)
        else:
            gate=id2
        array.append(gate)
        i+=1
    hadamade=qt.Qobj([[1]])
    for gate in array:
        hadamade = qt.tensor(hadamade,gate)
        hadamade=tensor_fix(hadamade)
    return hadamade

def main():
    
    pop=input("How many of each gate do you want to populate? ")
    pop=int(pop)
    split=input("Give training set split, in the form of a number between 0 and 1: ")
    k =input("Give a k value: ")
    k=int(k)
    d=input("Gimme a range of reference states: ")
    d=int(d)
    qubits=4
    n=2**qubits
    csvpath=["GHZTrainingData200new.csv","GHZTrainingData200QFT.csv","GHZTrainingData200Had.csv","GHZTrainingData200QFT2.csv"]
    #special subset for quantum repeater
    ghzcirc=[hadamaker(qubits,[0]),qt.cnot(qubits,0,1),qt.cnot(qubits,1,2),qt.cnot(qubits,2,3)]
    ghz_tags=["Hadamard","CNOT","CNOT","CNOT"]
    circuit=[]
    angles=[]
    for i in range(len(ghzcirc)):
        circuit.append(ghzcirc[i])
        circuit.append(ghz_tags[i])
    cat=categorize(circuit)
    alt=[]
    angles=[]
    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        else:
            alt.append(gate_troubleshooter(circuit[i],n))
    choice = [2,4,5,6]
    vector_name = ['new','QFT','Hadamard 2','Fourier State']
    index = 0
    for x in choice:
        choice = x
        path=csvpath[index]
        vectors=gen_basis_vectors(n,n,choice)
        print(vector_name[index])
        index = index+1
        probs=colin_mochrie(alt,angles,vectors,pop,cat,qubits,d,path)
        KNN(probs,split,k)
    return 0

start = time.time()
main()
print(time.time()-start)
   


# In[ ]:





# In[2]:


#ToleranceTest
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:05:41 2020

@author: marga
"""

import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
from qutip.qip.algorithms import qft
"""
KNN Block
"""

#splits the data set into testing and training data. Need to initialize empty vectors first
def handleDataset(array,split,trainingSet=[],testSet=[]):
    dataset=array
    for x in range(len(dataset)-1):
        for y in range(len(dataset[0])-1):
            dataset[x][y]=float(dataset[x][y])
            if random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
    return 0

#Finds the euclidean distance bewteen two vectors of length 'length;
def euclideanDistance(ins1,ins2,length):
    dis=0
    for x in range(length):
        dis += pow((ins1[x]-ins2[x]),2)
    return math.sqrt(dis)

#finds the k points nearest a test point
def getKNeighbors(trainingSet,test,k):
    distances=[]
    length=len(test)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(test,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#determines the classes of a vector of neighbors and returns a prediction of a test point's class
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
        sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]
    
#Finds the accuracy of a test set prediction   
def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
        #if testSet[x][-1] != predictions[x]:
            #print("predicted:" + predictions[x],"actual:" + testSet[x][-1])
    return (correct/float(len(testSet)))*100.0
    
def KNN(path,split,k):
    testingset=[]
    trainingset=[]
    #Insert csv file name and split here:
    #path=input("Give csv file location, remembering to use forward slashes: ")
    split = split
    split=float(split)
    if split > 1 or split < 0:
        print("Incorrect split input given. Default used.")
        split = 0.66
    handleDataset(path,split,trainingset,testingset)
    # generate predictions 
    predictions=[] 
    k=k
    for x in range(len(testingset)): 
        neighbors = getKNeighbors(trainingset,testingset[x],k) 
        result = getResponse(neighbors) 
        predictions.append(result) 
        print('> predicted=' + repr(result) + ',actual=' + repr(testingset[x][-1]))
    accuracy = getAccuracy(testingset,predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
###############################################################################
"""
Basic building-block functions for classification procedure
"""

#Fixes qutip's weird dimension problems
def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result

id2=qt.Qobj(np.identity(2))

#applies an array of gates (circuit) and applies it to a state
def basic_b(state,array):
    for i in range(len(array)):
        if type(array[i]) is str:
            continue
        else:
            k=qt.Qobj(array[i])
            #print(type(k))
            k=tensor_fix(k)
            #print(k.type,k.shape)
            state=qt.Qobj(state)
            #print(state)
            state=k(state)
    return state
    
#Fidelity distance metric
def dis(state1,state2):
    (n,m)=state1.shape
    if state1.type is not 'ket' or state2.type is not 'ket':
        print("Error: One or more input to the distance function is not a ket." )
        p_0=0
    else:
        fid=qt.fidelity(state1,state2)
        p_0 = 1/n + (n-1)*fid/n
    return p_0

def gate_troubleshooter(gate,n):
    if gate.shape != (n,n):
        gate=tensor_fix(gate)
        while gate.shape !=(n,n):
            seed=randint(0,1)
            if seed ==1:
                gate=qt.tensor(id2,gate)
                gate=tensor_fix(gate)
            else:
                gate=qt.tensor(gate,id2)
                gate=tensor_fix(gate)
        if gate.shape != (n,n):
            print("FUUUUUUUUUUUUUUUUU")
    return gate

"""
Hadamard generation gates
"""

#messed up hadamard
def multi_qubit_hadamard(regular_hadamard_gate):
    theta = uniform(0.0,math.pi*2.0)
    (n,n) = regular_hadamard_gate.shape
    N = np.int(((np.log(n))/(np.log(2))))
    phase=qt.globalphase(theta,N)
    phase=tensor_fix(phase)
    reg=tensor_fix(regular_hadamard_gate)
    multi_qubit_hadamard = phase*reg
    multi_qubit_hadamard = tensor_fix(multi_qubit_hadamard)    
    return multi_qubit_hadamard

#Looks awful, I promise it isn't. while loops are mostly security
#Just figures out whether there's a hadamard present we can alter
def hadamard_preprocessing(hada):
    storage=hada.full()
    (n,n)=hada.shape
    q=np.log(n)/np.log(2) #number of qubits
    seed=randint(0,q-1)
    forbidden=[] #a vector to hold forbidden seeds
    mag=storage[0][0] #magnitude of the elements in the hadamard
    ongoing=True
    while ongoing:
        i=1
        count=0
        while seed in forbidden: #make sure the seed isn't forbidden
            seed=randint(0,q-1)
            count+=1
            if count == 20: #no infinite loops
                break
        #initialize test unitary
        if seed == 0:
            u1=qt.hadamard_transform(1)
        else:
            u1=id2
        #create test unitary
        while u1.shape != (n,n):
            if i ==seed: #set a hadamard on specified qubit
                u1=qt.tensor(u1,qt.hadamard_transform(1))
                u1=tensor_fix(u1)
            else:
                u1=qt.tensor(u1,id2)
                u1=tensor_fix(u1)
            i+=1
        check=u1*hada
        if check.full()[0][0] > mag: #if there's a hadamard on that qubit, true
            ongoing = False
        elif count == 20:
            print("oops")
            break
        else: #no hadamard on that seed qubit
            forbidden.append(seed)
    return hada,seed

#can feed as input the preprocessing step
def alter_hadamard(hada,seed):
    (n,n)=hada.shape
    theta = uniform(0.0,math.pi*2.0)
    #pick a rotation any rotation
    phaser=randint(0,3)
    if phaser ==0:
        gate=qt.phasegate(theta)
    elif phaser == 1:
        gate=qt.rz(theta)
    elif phaser==2:
        gate = qt.ry(theta)
    else:
        gate=qt.globalphase(theta)
    #alter gate
    if seed == 0:
        u1=gate
    else:
        u1=id2
    i=1
    while u1.shape != (n,n):
        if i ==seed: #set a alteration on specified qubit
            u1=qt.tensor(u1,gate)
            u1=tensor_fix(u1)
        else:
            u1=qt.tensor(u1,id2)
            u1=tensor_fix(u1)
        i+=1
    final_gate=u1*hada
    return final_gate

def h_reassign(hada):
    seed = randint(0,1)
    if seed == 0: #alter whole hadamard
        alt_had=multi_qubit_hadamard(hada)
    if seed == 1: #alter specific hadamard
        (h,seed)=hadamard_preprocessing(hada)
        alt_had=alter_hadamard(hada,seed)
    return alt_had

#affect is list of affected qubits, 0 indexed
#makes an n qubit hadamard, affecting a subset of qubits
def hadamaker(qubits,affected):
    array=[]
    i=0
    while len(array) < qubits:
        if i in affected:
            gate=qt.hadamard_transform()
            gate=tensor_fix(gate)
        else:
            gate=id2
        array.append(gate)
        i+=1
    hadamade=qt.Qobj([[1]])
    for gate in array:
        hadamade = qt.tensor(hadamade,gate)
        hadamade=tensor_fix(hadamade)
    return hadamade

"""
Random Unitary gates/functions
"""
#code which give an original unitary gate
def random_unitary_gate(delta,alpha,theta,beta,value):
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
    if value == True:
        gate = gate *qt.Qobj([[0,1],[1,0]])
    else:
        gate = gate
    return gate

def random_angles():
    #gets a random value for each variable in the gate
    choice = randint(1,4)
    unitary_gate = ()
    if choice == 1: #Pauli-Y Gate
        unitary_gate = (0.0,math.pi/2,2*math.pi,math.pi/2,True)
    elif choice == 2: #Pauli-Z Gate
        unitary_gate = (0.0,0.0,math.pi,0.0,True)
    elif choice == 3: #S Gate
        unitary_gate = (-math.pi/2,math.pi,math.pi,math.pi,False)
    elif choice == 4: #T Gate
        unitary_gate = (-math.pi/4,math.pi/2,2*math.pi,0.0,False)
    delta,alpha,theta,beta,value = unitary_gate
    return delta,alpha,theta,beta,value

#code which takes an angle and alters the gate
def random_altered_unitary_gate(delta,alpha,theta,beta,value):
    if delta == 0.0 and alpha == 0.0 and theta == math.pi and value == True:
        angles = ['delta','alpha','beta']
    else:
        angles = ['delta','alpha','theta','beta']
    altered_variable = choice(angles)
    if altered_variable == 'delta':
        delta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'alpha':
        alpha = uniform(0.0,2.0*math.pi)
    if altered_variable == 'theta':
        theta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'beta':
        beta = uniform(0.0,2.0*math.pi)
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
    if value == True:
        gate = gate *qt.Qobj([[0,1],[1,0]])
    else:
        gate = gate
    return gate
   
#gives both an original and altered unitary gate
#can be commented to return oritinal gate, corruspondig altered gate(onle one thing different from original), or both
def unitary_gate(choice):
    delta,alpha,theta,beta,value = random_angles()
    original = random_unitary_gate(delta,alpha,theta,beta,value)
    matrix=original
    if choice:
        altered = random_altered_unitary_gate(delta,alpha,theta,beta,value)
        matrix=altered
    return (matrix,[delta,alpha,theta,beta,value])

"""
CNOT & CZ functions
"""
#alter a CNOT
def rot(qubits,choice):
    a=randint(0,qubits-2)
    b=randint(a+1,qubits-1)
    k=qt.cnot(qubits,a,b)
    k=k.full()
    if choice:
        k=np.random.permutation(k)
        k=qt.Qobj(k)
        #k=k*k.dag()
        k=tensor_fix(k)
    cn_final=qt.Qobj(k)
    return cn_final

def conv_cz():
    gate=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,-1]])
    gate=qt.Qobj(gate)
    return gate

"""
Classification and Alteration Protocols
"""
def categorize(circuit):
    cat=[]
    it_not=1
    it_h=1
    it_rand=1
    it_id=0
    composition=[]
    
    
    for i in range(len(circuit)):
        if i % 2 != 0:
            continue
        if circuit[i+1] in cat:
            if "Hadamard" in circuit[i+1]:
                it_h+=1
                t=str(it_h)
                t="Hadamard" + t
                cat.append(t)
            elif "CNOT" in circuit[i+1]:
                it_not+=1
                t=str(it_not)
                t="CNOT" + t
                cat.append(t)
            elif "Random Unitary" in circuit[i+1]:
                it_rand+=1
                t=str(it_rand)
                t="Random Unitary" + t
                cat.append(t)
        elif "Identity" in circuit[i+1]:
            it_id +=1
            t=str(it_id)
            t="Measurement" + t
            cat.append(t)
        else:
            cat.append(circuit[i+1])
        i+=2
    return(cat)
    
def rabbit(qubits,choice,start):
    basic_0ket=qt.Qobj([[1],[0]])
    basic_1ket=qt.Qobj([[0],[1]])
    if start ==1:
        temp=basic_1ket
    else:
        temp=basic_0ket
    r=1
    while r < qubits:
        if r in choice:
            temp=qt.tensor(temp,basic_1ket)
            temp=tensor_fix(temp)
        else:
            temp=qt.tensor(temp,basic_0ket)
            temp=tensor_fix(temp)
        r+=1
    return temp
    
def gen_basis_vectors(n,dims,choice):
    vectors=[]
    basic_states=[]
    bits=int(math.log(n,2))
    #for i in range(n):
        #state=qt.basis(n,i)
        #fock_states.append(state)
    q=rabbit(bits,[],0)
    basic_states.append(q)
    q=rabbit(bits,[1],0)
    basic_states.append(q)
    q=rabbit(bits,[n-1],0)
    basic_states.append(q)
    indexvec=[]
    for i in range(bits):
        indexvec.append(i)
    q=rabbit(bits,indexvec,1)
    basic_states.append(q)
    test_opt1=[basic_states[0],basic_states[1],basic_states[2],basic_states[3]]
    test_opt2=[basic_states[0],basic_states[1],basic_states[2],basic_states[3],basic_states[-1]+basic_states[1]]
    if choice == 1: #Basis states
        vectors = basic_states
        q=qt.Qobj(np.ones(n))
        q=q.unit()
        vectors.append(q)
    elif choice ==2:
        vectors=basic_states
    elif choice == 3: #Hadamard option 1
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 4: #QFT option 1
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 5: #Hadamard option 2
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 6: #QFT option 2
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
        vectors.append(state_n)
    return vectors

"""
"""
def get_ideal(circuit,vectors,qubits,d): 
    probabilities=[]
    n=2**qubits
    references=[]
    for chi in range(d):
        compare=gen_basis_vectors(n,n,4)
        references.append(compare[chi])
    temparray=[]
    i=0
    for ref in references:
        state=vectors[i]
        final=basic_b(state,circuit)
        prob=dis(final,ref)
        temparray.append(prob)
        i+=1
    probabilities.append(temparray)
    return probabilities



def within_tolerance(tolerance,probvector,idealvector):
    length=4
    scale=1-tolerance
    test=scale*euclideanDistance([0,0,0,0],idealvector,length)
    dist=euclideanDistance(probvector,idealvector,length)
    if dist <= test:
        truth = True
    else:
        truth = False
    return truth

def main():
    d=4
    qubits=3
    n=2**qubits
    state_creator=[hadamaker(qubits,[1]),qt.cnot(qubits,1,2),qt.cnot(qubits,0,1),hadamaker(qubits,[0]),qt.cnot(qubits,1,2),conv_cz()]
    state_creator_tags=["Hadamard","CNOT","CNOT2","Hadamard2","CNOT3","Control Z"]
    circuit=[]

    for i in range(len(state_creator)):
        circuit.append(state_creator[i])
        circuit.append(state_creator_tags[i])
    alt=[]
    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        else:
            alt.append(gate_troubleshooter(circuit[i],n))
    vectors=gen_basis_vectors(n,n,2)
    ideal=get_ideal(alt,vectors,qubits,d)
    print(ideal[0])
    tolerance=0.9
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    multi=uniform(0,1)
    ideal2=ideal[0]
    probvector=[]
    for x in range(len(ideal2)):
        probvector.append(multi*ideal2[x])
    truth=within_tolerance(tolerance,probvector,ideal[0])
    print(multi)
    print(truth)
    return 0
    
main()

