# https://github.com/Dylan-Nico/Quantum-1-Nearest-Neighour/blob/4ac521409a6aef27f400f89b1b533d14e2c35fe7/Circuiti%20con%20Banknote%20dataset/64-Q1NN.py
#!/usr/bin/env python
# coding: utf-8

# In[16]:


import qiskit
import qiskit.quantum_info as qi
from qiskit.visualization import plot_bloch_vector
from qiskit.circuit.library import RYGate, MCMT, RYGate, CRYGate
from sklearn import preprocessing

import csv
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


# In[17]:


data = pd.read_csv("bank.txt",header=None,names=["f0","f1","f2","f3","class"])


# In[18]:


#Standardise
scaler = StandardScaler()
data.loc[:,["f0","f1","f2","f3"]] = scaler.fit_transform(data.loc[:,["f0","f1","f2","f3"]])

#Normalize
data.loc[:,["f0","f1","f2","f3"]] = preprocessing.normalize(data.loc[:,["f0","f1","f2","f3"]], axis=1)


# In[19]:


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


# In[20]:


# random_seed : int : Random number generator seed
random_seed = 3
rgen = np.random.RandomState(random_seed)
def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]


# In[21]:


data = data.iloc[rgen.permutation(len(data.index))].copy()
data = data[347:]
data_copy = data
data_copy


# In[22]:


def encodeClasse(circuit,classe,t):
    if classe == 0:
        circuit.x(t)

def encodeIndex(circuit,index,u):
    getBinary = lambda x, n: format(x, 'b').zfill(n)
    index = getBinary(index,6)
    #inverto la stringa
    index = index[::-1]
    for l in range(len(index)):
        if(index[l] == '0'):
            circuit.x(u[l])


# In[23]:


#creazione registri
prova = QuantumRegister(1,"p") #fidelity ancilla
i = QuantumRegister(2,"i")
j = QuantumRegister(2,"j")  #sempre 2 per indicare le features. Sono fissi.
q = QuantumRegister(6,"q") #qbit indice per i training
r = QuantumRegister(2,"r") 
classe = QuantumRegister(1,"classe") #2 qbits --> quattro classi (a noi ne servono 3)
b = ClassicalRegister(1,"b") #for measure fideilty ancilla
c = ClassicalRegister(2,"c") #for measure r0r1, has to be both 11 on the histogram
c3 = ClassicalRegister(1,"c3") #for measure qbit classi 00,01,10
c4= ClassicalRegister(6,"c4") #for measure q0q1q2 indexes


# In[24]:


risultati = []
ground_truth = []
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
        subset = data.iloc[:64]
        data = data.drop(subset.index)
        print("Circuito:",p)
        print(subset)
    
        #creazione circuito
        circuit.append(QuantumCircuit(prova,i,j,r,q,classe,b,c,c4,c3))
        circuit[p].h(i)
        circuit[p].h(j)
        circuit[p].h(q)
        circuit[p].h(classe)
        #encode inputvector
        circuit[p].barrier()
        encodeVector(circuit[p], inputVector, i, i[:], r[0],None) 
        circuit[p].barrier()
        #encode training
        limit = len(subset)
        for k in range(limit):
            trainingVector = subset.iloc[k]
            encodeClasse(circuit[p],trainingVector["class"],classe)
            encodeIndex(circuit[p],k,q)
            encodeVector(circuit[p], trainingVector["f0":"f3"], j, [q[0], q[1], q[2],q[3],q[4],q[5], j[0],j[1],classe[0]], r[1], None)
            encodeIndex(circuit[p],k,q)
            encodeClasse(circuit[p],trainingVector["class"],classe)
        #fidelity
        circuit[p].h(prova[0])
        circuit[p].cswap(prova[0],i[0],j[0])
        circuit[p].cswap(prova[0],i[1],j[1])
        circuit[p].cswap(prova[0],r[0],r[1])
        circuit[p].h(prova[0])
        circuit[p].barrier()
        #misurazioni
        circuit[p].measure(prova[0],b[0])
        circuit[p].measure(r[0],c[0])
        circuit[p].measure(r[1],c[1])
        circuit[p].measure(q[0],c4[0])
        circuit[p].measure(q[1],c4[1])
        circuit[p].measure(q[2],c4[2])
        circuit[p].measure(q[3],c4[3])
        circuit[p].measure(q[4],c4[4])
        circuit[p].measure(q[5],c4[5])
        circuit[p].measure(classe[0],c3[0])
        #result
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit[p], simulator, shots=60000000)
        result = job.result()
        counts = result.get_counts(circuit[p])
        risultati.append(counts)
        p+=1
    data = data_copy #ripristino il dataset prima della prossima iterazione


# In[30]:


len_risultati = len(risultati)
final = []
for w in range(len_risultati):
    goodResult = risultati[w]
    
    #ordino le configurazioni
    sort_counts = sorted(goodResult.items())
    m = len(sort_counts)

    #lascio "libera" la classe
    goodValues = []
    for k in range(m):
        value = sort_counts[k][0]
        if(value[9] == '1' and value[10] == '1' and value[12] == '0'):
            goodValues.append(sort_counts[k])

    #converto in dizionario per il plot
    goodValues = dict((k, y) for k, y in goodValues) 
    final.append(goodValues)


# In[31]:


#estraggo il valore più alto per ognuno degli istogrammi 
prediction = []
lunghezza_final = len(final)
for i in range(lunghezza_final):
    predict = max(final[i], key=final[i].get)
    predict = str(predict)
    print("Circuito:",i,"Classe predetta:",int(predict[0]))
    prediction.append(int(predict[0]))


# In[ ]:



fields = ['#Vectors', 'Predition', 'Ground truth']
filename = "1QNN32_records_BankDataset.csv"
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)
    
    # writing the fields 
    csvwriter.writerow(fields)
    
    j = 0
    for i in range(len(ground_truth)):
        while j < len(prediction): 
            row = [['32',prediction[j],ground_truth[i]]]
            csvwriter.writerows(row)
            j+=1
            if ((j%8) == 0):
                break
            


# In[ ]:




