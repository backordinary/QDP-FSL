# https://github.com/Dylan-Nico/Quantum-1-Nearest-Neighour/blob/4ac521409a6aef27f400f89b1b533d14e2c35fe7/Circuiti%20con%20Iris%20dataset/2-Q1NN_StatePreparation.py
#!/usr/bin/env python
# coding: utf-8

# In[25]:


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


# In[26]:


iris = pd.read_csv("Iris/iris.data",header=None,names=["f0","f1","f2","f3","class"])


# In[27]:


#Standardise
scaler = StandardScaler()
iris.loc[:,["f0","f1","f2","f3"]] = scaler.fit_transform(iris.loc[:,["f0","f1","f2","f3"]])

#Normalize
iris.loc[:,["f0","f1","f2","f3"]] = preprocessing.normalize(iris.loc[:,["f0","f1","f2","f3"]], axis=1)


# In[28]:


setosa = iris[iris["class"] == "Iris-setosa"]
versicolor = iris[iris["class"] == "Iris-versicolor"]
virginica = iris[iris["class"] == "Iris-virginica"]

#dataset con 129 righe (128 quando sceglierò il test)
versicolor = versicolor.iloc[:-11,:]
virginica = virginica.iloc[:-10,:]


# In[29]:


data = pd.concat([setosa,versicolor,virginica])


# In[30]:


# random_seed : int : Random number generator seed
random_seed = 2
rgen = np.random.RandomState(random_seed)
def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]


# In[31]:


data = data.iloc[rgen.permutation(len(data.index))].copy()
data


# In[32]:


#prendo un input e lo tolgo dai dati di training
inputVector = data.loc[67]
data = data.drop(67)
inputVector


# In[33]:


# Il training array completo è composto da elementi per ogni classe
trainingArray = data.iloc[58:60]
trainingArray
data = data.drop(trainingArray.index)


# In[34]:


def encodeClasse(circuit,irisClass,c):
    classSwitcher = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    
    if classSwitcher.get(irisClass) == 0:
        circuit.x(c)
    elif classSwitcher.get(irisClass) == 1:
        circuit.x(c[1])
    elif classSwitcher.get(irisClass) == 2:
        circuit.x(c[0])

def encodeIndex(circuit,index,u):
    getBinary = lambda x, n: format(x, 'b').zfill(n)
    index = getBinary(index,2)
    #inverto la stringa
    index = index[::-1]
    for l in range(len(index)):
        if(index[l] == '0'):
            circuit.x(u[l])
        


# In[35]:


#n training -> log(n) per indicizzarli
n = math.log(len(trainingArray),2)
n = round(n) #prendo la parte intera superiore


# In[36]:


#creazione circuito
prova = QuantumRegister(1,"p")
q = QuantumRegister(1,"q") #qbit indice per i training
r = QuantumRegister(2,"r")
r2 = QuantumRegister(2,"r2")
classe = QuantumRegister(2,"classe") #2 qbits --> quattro classi (a noi ne servono 3)
b = ClassicalRegister(1,"b")
c = ClassicalRegister(2,"c")
c2 = ClassicalRegister(2,"c2")
c3 = ClassicalRegister(2,"c3") #for measure qbit classi 
c4 = ClassicalRegister(1,"c4")

circuit = QuantumCircuit(prova,r,r2,q,classe,b,c4,c3)

circuit.h(q) #creo le configurazioni per l'indicizzazione
circuit.h(classe) #hadamard per distinguere la classe 


# In[37]:


#encode test
circuit.barrier()
controlled_gate = StatePreparation(inputVector["f0":"f3"])  
circuit.append(controlled_gate,[r[0],r[1]])
circuit.barrier()


# In[38]:


#encode training vectors
'''
length = len(trainingArray)
print("Lunghezza training data: ",length)
for k in range(length):
    print("Indice: ", k)
    trainingVector = trainingArray.iloc[k]
    print("----START---- ")
    print(trainingVector)
    encodeClasse(circuit,trainingVector["class"],classe)
    encodeIndex(circuit,k,q)
    controlled_gate = StatePreparation(trainingVector["f0":"f3"]).control(4)  
    circuit.append(controlled_gate,[q[0],q[1],classe[0],classe[1],r2[0],r2[1]])
    encodeIndex(circuit,k,q)
    encodeClasse(circuit,trainingVector["class"],classe)
    print("----END----")
'''
trainingVector = trainingArray.iloc[0]
encodeClasse(circuit,trainingVector["class"],classe)
circuit.x(q)
controlled_gate = StatePreparation(trainingVector["f0":"f3"]).control(3)
circuit.append(controlled_gate,[q[0],classe[0],classe[1],r2[0],r2[1]])
circuit.x(q)
encodeClasse(circuit,trainingVector["class"],classe)

trainingVector2 = trainingArray.iloc[1]
encodeClasse(circuit,trainingVector2["class"],classe)
controlled_gate = StatePreparation(trainingVector2["f0":"f3"]).control(3)
circuit.append(controlled_gate,[q[0],classe[0],classe[1],r2[0],r2[1]])
encodeClasse(circuit,trainingVector2["class"],classe)

circuit.barrier()


# In[39]:


circuit.draw("mpl")


# In[40]:


#la probabilità esce 1 quando sono uguali (e avviene quando il bit di misurazione è 0)
#non c'è la configurazione zero perchè la fidelity esce 0 ?

#fidelity
#sommario: in r[0] c'è il test, in r[1] tr0 e tr1 in superposition
circuit.h(prova[0])
circuit.cswap(prova[0],r[0],r2[0])
circuit.cswap(prova[0],r[1],r2[1])
circuit.h(prova[0])
circuit.barrier()
circuit.measure(prova[0],b[0])
circuit.measure(q[0],c4[0])
circuit.measure(classe[0],c3[0])
circuit.measure(classe[1],c3[1])


# In[41]:


#circuit.draw('mpl')


# In[42]:


## results
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=200000)
result = job.result()
counts = result.get_counts(circuit)
plot_histogram(counts)


# In[43]:


#POST SELECTION
goodCounts = {k: counts[k] for k in counts.keys() & {'01 0 0', '10 1 0'}}
plot_histogram(goodCounts) 


# In[21]:


from qiskit.compiler import transpile


backend = Aer.get_backend('qasm_simulator')
#backend = provider.get_backend('ibmq_16_melbourne')
transpile_circuit = transpile(circuit, backend)
#transpile_circuit.draw('mpl')


# In[23]:


print(transpile_circuit.depth())


# In[24]:


transpile_circuit.draw('mpl')


# In[ ]:




