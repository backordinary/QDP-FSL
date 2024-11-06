# https://github.com/Dylan-Nico/Quantum-1-Nearest-Neighour/blob/4ac521409a6aef27f400f89b1b533d14e2c35fe7/Circuiti%20con%20Iris%20dataset/8-Q1NN_StatePreparation.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit
import qiskit.quantum_info as qi
from qiskit.visualization import plot_bloch_vector
from qiskit.circuit.library import RYGate, MCMT, RYGate, CRYGate,StatePreparation
from qiskit.extensions import *
from qiskit.quantum_info import Operator
from sklearn import preprocessing

import numpy as np
import pandas as pd
import math
from numpy import linalg,dot
from qiskit import IBMQ
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer)
from qiskit.visualization import plot_histogram
from sklearn.preprocessing import StandardScaler


# In[2]:


iris = pd.read_csv("Iris/iris.data",header=None,names=["f0","f1","f2","f3","class"])


# In[3]:


#Standardise
scaler = StandardScaler()
iris.loc[:,["f0","f1","f2","f3"]] = scaler.fit_transform(iris.loc[:,["f0","f1","f2","f3"]])

#Normalize
iris.loc[:,["f0","f1","f2","f3"]] = preprocessing.normalize(iris.loc[:,["f0","f1","f2","f3"]], axis=1)


# In[4]:


setosa = iris[iris["class"] == "Iris-setosa"]
versicolor = iris[iris["class"] == "Iris-versicolor"]
virginica = iris[iris["class"] == "Iris-virginica"]


# In[5]:


#dataset con 129 righe (128 quando sceglierò il test)
versicolor = versicolor.iloc[:-11,:]
virginica = virginica.iloc[:-10,:]


# In[6]:


data = pd.concat([setosa,versicolor,virginica])


# In[7]:


# random_seed : int : Random number generator seed
random_seed = 3
rgen = np.random.RandomState(random_seed)
def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]


# In[8]:


data = data.iloc[rgen.permutation(len(data.index))].copy()
data_copy = data
#data_copy


# In[9]:


def encodeClasse(circuit,irisClass,t):
    classSwitcher = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    
    if classSwitcher.get(irisClass) == 0:
        circuit.x(t)
    elif classSwitcher.get(irisClass) == 1:
        circuit.x(t[1])
    elif classSwitcher.get(irisClass) == 2:
        circuit.x(t[0])

def encodeIndex(circuit,index,u):
    getBinary = lambda x, n: format(x, 'b').zfill(n)
    index = getBinary(index,3)
    #inverto la stringa
    index = index[::-1]
    for l in range(len(index)):
        if(index[l] == '0'):
            circuit.x(u[l])


# In[10]:


#creazione registri
prova = QuantumRegister(1,"p") #fidelity ancilla
q = QuantumRegister(3,"q") #qbit indice per i training
r = QuantumRegister(2,"r")
r2 = QuantumRegister(2,"r2")
classe = QuantumRegister(2,"classe") #2 qbits --> quattro classi (a noi ne servono 3)
b = ClassicalRegister(1,"b") #for measure fideilty ancilla
c3 = ClassicalRegister(2,"c3") #for measure qbit classi 00,01,10
c4= ClassicalRegister(3,"c4") #for measure q0q1 indexes


# In[11]:


risultati = []
ground_truth = []
sub = []
for v in data.index:
    p = 0
    print("Indice for loop:",v)
    circuit = []
    inputVector = data.loc[v]
    data = data.drop(v)
    ground_truth.append(inputVector["class"])
    print("Classe inputVector:",inputVector["class"])
    print("Riga inputVector:",inputVector)
    while not data.empty:
        #estrazione del sottoinsieme
        subset = data.iloc[:8]
        sub.append(subset)
        data = data.drop(subset.index)
        print("Circuito:",p)
        print(subset)
    
        #creazione circuito
        circuit.append(QuantumCircuit(prova,r,r2,q,classe,b,c4,c3))
        circuit[p].h(q)
        circuit[p].h(classe)
        #encode inputvector
        circuit[p].barrier()
        controlled_gate = StatePreparation(inputVector["f0":"f3"]) 
        circuit[p].append(controlled_gate,[r[0],r[1]]) 
        circuit[p].barrier()
        #encode training
        limit = len(subset)
        for k in range(limit):
            trainingVector = subset.iloc[k]
            encodeClasse(circuit[p],trainingVector["class"],classe)
            encodeIndex(circuit[p],k,q)
            controlled_gate = StatePreparation(trainingVector["f0":"f3"]).control(5)
            circuit[p].append(controlled_gate,[q[0],q[1],q[2],classe[0],classe[1],r2[0],r2[1]])
            encodeIndex(circuit[p],k,q)
            encodeClasse(circuit[p],trainingVector["class"],classe)
        #fidelity
        circuit[p].h(prova[0])
        circuit[p].cswap(prova[0],r[0],r2[0])
        circuit[p].cswap(prova[0],r[1],r2[1])
        circuit[p].h(prova[0])
        circuit[p].barrier()
        #misurazioni
        circuit[p].measure(prova[0],b[0])
        circuit[p].measure(q[0],c4[0])
        circuit[p].measure(q[1],c4[1])
        circuit[p].measure(q[2],c4[2])
        circuit[p].measure(classe[0],c3[0])
        circuit[p].measure(classe[1],c3[1])
        #result
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit[p], simulator, shots=5000000)
        result = job.result()
        counts = result.get_counts(circuit[p])
        risultati.append(counts)
        p+=1
    data = data_copy #ripristino il dataset prima della prossima iterazione


# In[12]:


tmp = []
total = [] #total[i] conterrà le stringhe degli output corretti dell'i-esimo training set 
N = int(math.log(len(subset),2))
print(N)
for x in sub:
    tr = x
    for y in range(len(tr)):
        getBinary = lambda x, n: format(x, 'b').zfill(n)
        f = tr.iloc[y]
        if(f["class"] == "Iris-setosa"):
            tmp.append('00 ' + getBinary(y,N) + ' 0')
        if(f["class"] == "Iris-versicolor"):
            tmp.append('01 ' + getBinary(y,N) + ' 0')
        if(f["class"] == "Iris-virginica"):
            tmp.append('10 ' + getBinary(y,N) + ' 0')
    total.append(tmp)
    tmp = []
    
#-----post-selection required!------
#extract predictions
for k in range(len(risultati)):
    tmp2 = risultati[k]
    goodCounts = {k: tmp2[k] for k in tmp2.keys() & total[k]}
    predict = max(goodCounts, key=goodCounts.get)
    predict = str(predict)
    print("Circuito:",k,"Classe predetta:",int(predict[:2],2))
    #prediction.append(int(predict[:2],2))



# In[ ]:


#profondità e dimensione

from qiskit.compiler import transpile


backend = Aer.get_backend('qasm_simulator')
#backend = provider.get_backend('ibmq_16_melbourne')
transpile_circuit = transpile(circuit[0], backend)
#transpile_circuit.draw('mpl')

print(transpile_circuit.depth())
print(transpile_circuit.size())


# In[ ]:




