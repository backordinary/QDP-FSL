# https://github.com/Dylan-Nico/Quantum-1-Nearest-Neighour/blob/4ac521409a6aef27f400f89b1b533d14e2c35fe7/Circuiti%20con%20Iris%20dataset/2-Q1NN_QRAM.py
#!/usr/bin/env python
# coding: utf-8

# In[71]:


import qiskit
import qiskit.quantum_info as qi
from qiskit.visualization import plot_bloch_vector
from qiskit.circuit.library import RYGate, MCMT, RYGate, CRYGate
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
from operator import itemgetter
from sklearn.preprocessing import StandardScaler


# In[72]:


iris = pd.read_csv("Iris/iris.data",header=None,names=["f0","f1","f2","f3","class"])


# In[73]:


#Standardise
scaler = StandardScaler()
iris.loc[:,["f0","f1","f2","f3"]] = scaler.fit_transform(iris.loc[:,["f0","f1","f2","f3"]])

#Normalize
iris.loc[:,["f0","f1","f2","f3"]] = preprocessing.normalize(iris.loc[:,["f0","f1","f2","f3"]], axis=1)


# In[74]:


setosa = iris[iris["class"] == "Iris-setosa"]
versicolor = iris[iris["class"] == "Iris-versicolor"]
virginica = iris[iris["class"] == "Iris-virginica"]


# In[75]:


#dataset con 129 righe (128 quando sceglierò il test)
versicolor = versicolor.iloc[:-11,:]
virginica = virginica.iloc[:-10,:]


# In[76]:


data = pd.concat([setosa,versicolor,virginica])


# In[77]:


#qram
def encodeVector(circuit,data,i,controls,rotationQubit,ancillaQubits):
    #mcry(angolo,controls,rotation,ancilla)
    
    # |00>
    circuit.x(i[0])
    circuit.x(i[1])
    circuit.append(MCMT(CRYGate(2*np.arcsin(data[0])), num_ctrl_qubits=len(controls), num_target_qubits=1), controls[0:]+[rotationQubit])
    #circuit.mcry(np.arcsin(data[0]),controls,rotationQubit,ancillaQubits)
    circuit.x(i[0])
    circuit.x(i[1])
    
    circuit.barrier()
    # |01>
    circuit.x(i[1])
    circuit.append(MCMT(CRYGate(2*np.arcsin(data[1])), num_ctrl_qubits=len(controls), num_target_qubits=1), controls[0:]+[rotationQubit])
    #circuit.mcry(np.arcsin(data[1]),controls,rotationQubit,ancillaQubits)
    circuit.x(i[1])
    
    circuit.barrier()
    # |10>
    circuit.x(i[0])
    circuit.append(MCMT(CRYGate(2*np.arcsin(data[2])), num_ctrl_qubits=len(controls), num_target_qubits=1), controls[0:]+[rotationQubit])
    #circuit.mcry(np.arcsin(data[2]),controls,rotationQubit,ancillaQubits)
    circuit.x(i[0])
    
    circuit.barrier()
    # |11>
    circuit.append(MCMT(CRYGate(2*np.arcsin(data[3])), num_ctrl_qubits=len(controls), num_target_qubits=1), controls[0:]+[rotationQubit])
    #circuit.mcry(np.arcsin(data[3]),controls,rotationQubit,ancillaQubits)


# In[78]:


# random_seed : int : Random number generator seed
random_seed = 2
rgen = np.random.RandomState(random_seed)
def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]


# In[79]:


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


# In[80]:


data = data.iloc[rgen.permutation(len(data.index))].copy()


# In[81]:


#prendo un input e lo tolgo dai dati di training
inputVector = data.loc[67]
data = data.drop(67)
inputVector


# In[82]:


# Il training array completo è composto da elementi per ogni classe
trainingArray = data.iloc[58:60]
trainingArray


# In[83]:


data = data.drop(trainingArray.index)


# In[84]:


#creazione registri
prova = QuantumRegister(1,"p") #fidelity ancilla
i = QuantumRegister(2,"i")
j = QuantumRegister(2,"j")  #sempre 2 per indicare le features. Sono fissi.
q = QuantumRegister(1,"q") #qbit indice per i training
r = QuantumRegister(2,"r") 
classe = QuantumRegister(2,"classe") #2 qbits --> quattro classi (a noi ne servono 3)
b = ClassicalRegister(1,"b") #for measure fideilty ancilla
c = ClassicalRegister(2,"c") #for measure r0r1, has to be both 11 on the histogram
c3 = ClassicalRegister(2,"c3") #for measure qbit classi 00,01,10
c4= ClassicalRegister(1,"c4") #for measure q0q1 indexes

circuit = QuantumCircuit(prova,i,j,r,q,classe,b,c,c4,c3)


# In[85]:


circuit.h(classe)
circuit.h(q)
circuit.h(i)
circuit.h(j)
encodeVector(circuit, inputVector, i, i[:], r[0],None)

trainingVector = trainingArray.iloc[0]
encodeClasse(circuit,trainingVector["class"],classe)
circuit.x(q)
encodeVector(circuit, trainingVector["f0":"f3"], j, [q[0], j[0],j[1],classe[1],classe[0]], r[1], None)
circuit.x(q)
encodeClasse(circuit,trainingVector["class"],classe)

trainingVector2 = trainingArray.iloc[1]
encodeClasse(circuit,trainingVector2["class"],classe)
encodeVector(circuit, trainingVector2["f0":"f3"], j, [q[0], j[0],j[1],classe[1],classe[0]], r[1], None)
encodeClasse(circuit,trainingVector2["class"],classe)
circuit.barrier()
#fidelity
circuit.h(prova[0])
circuit.cswap(prova[0],i[0],j[0])
circuit.cswap(prova[0],i[1],j[1])
circuit.cswap(prova[0],r[0],r[1])
circuit.h(prova[0])
circuit.barrier()
#misurazioni
circuit.measure(prova[0],b[0])
circuit.measure(r[0],c[0])
circuit.measure(r[1],c[1])
circuit.measure(q[0],c4[0])
circuit.measure(classe[0],c3[0])
circuit.measure(classe[1],c3[1])


# In[86]:


#circuit.draw(output='mpl')


# In[87]:


#result
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=200000)
result = job.result()
counts = result.get_counts(circuit)


# In[88]:


plot_histogram(counts)


# In[89]:


#POST SELECTION
#ordino le configurazioni
sort_counts = sorted(counts.items())
m = len(sort_counts)

#lascio "libera" la classe
goodValues = []
for k in range(m):
    value = sort_counts[k][0]
    if(value[5] == '1' and value[6] == '1' and value[8] == '0'):
        goodValues.append(sort_counts[k])

#converto in dizionario per il plot
goodValues = dict((k, y) for k, y in goodValues) 


# In[90]:


plot_histogram(goodValues)


# In[91]:


from qiskit.compiler import transpile


backend = Aer.get_backend('qasm_simulator')
#backend = provider.get_backend('ibmq_16_melbourne')
transpile_circuit = transpile(circuit, backend)
#transpile_circuit.draw('mpl')


# In[92]:


#transpile_circuit.draw(output='mpl')


# In[93]:


print(transpile_circuit.depth())


# In[ ]:




