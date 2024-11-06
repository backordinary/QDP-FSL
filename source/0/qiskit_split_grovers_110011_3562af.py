# https://github.com/xaviermccaig/Quantum-Thesis/blob/dbfb8fc762db2a34230152e7b1b603bcd898ffd0/Qiskit_Split_Grovers_110011.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Qiskit libraries
get_ipython().run_line_magic('matplotlib', 'inline')
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator
from ibm_quantum_widgets import *
from qiskit.tools.monitor import job_monitor
from qiskit.quantum_info.operators import Operator

# General Imports
import numpy as np
from math import sqrt,log
warnings.filterwarnings('ignore')

# Loading IBMQ account
provider = IBMQ.load_account()


# In[2]:


############################################
# This function acts as the phase shift component of the Grover 
# iteration. A general diffuser, it needs only the number of index
# and total qubits.
############################################
def diffuser(qb,total):
    qc = QuantumCircuit(total)
    for qubit in range(qb):
        qc.h(qubit)
    
    for qubit in range(qb):
        qc.x(qubit)
        
    qc.x(total-1)
    qc.h(total-1)
    qc.mct([*range(qb)], total-1)  # multi-controlled Toffoli
    qc.h(total-1)
    qc.x(total-1)
    
    for qubit in range(qb):
        qc.x(qubit)
    
    for qubit in range(qb):
        qc.h(qubit)
        
    display(qc.draw('mpl'))
    Diffuser = qc.to_gate()
    Diffuser.name = "Diffuser"
    return Diffuser
############################################
# Initializes the qubits into superposition
############################################
def initialize(qc, qubits):
    for q in qubits:
        qc.h(q)
    return qc
############################################
# Creates the oracles to recognize each
# half of the index sought for, `110011'.
############################################
def oracle_maker(qb, half):
    # Creating oracles on the basis of which
    # half of the total index size is being searched
    if half == 1:
        ##110
        qc = QuantumCircuit(qb)
        qc.ccx(0,1,3)
        qc.x(4)
        qc.cx(2,4)
        qc.ccx(3,4,5)
        qc.z(5)
        qc.ccx(3,4,5)
        qc.cx(2,4)
        qc.x(4)
        qc.ccx(0,1,3)
        ##110
    if half == 2:
        ##011
        qc = QuantumCircuit(qb)
        qc.ccx(1,2,3)
        qc.x(4)
        qc.cx(0,4)
        qc.ccx(3,4,5)
        qc.z(5)
        qc.ccx(3,4,5)
        qc.cx(0,4)
        qc.x(4)
        qc.ccx(1,2,3)
        ##011
    
    display(qc.draw('mpl'))
    oracle = qc.to_gate()
    oracle.name = "Oracle"
    
    return oracle
############################################
# Main section
############################################
n_qb = 6 #n_qb = n
total = 2*n_qb
val = '110011'
sim = True
if sim is True:
    #backendOne = Aer.get_backend('qasm_simulator')
    #backendTwo = Aer.get_backend('qasm_simulator') 
    backendOne = QasmSimulator.from_backend(provider.backends.ibmq_16_melbourne)
    backendTwo = QasmSimulator.from_backend(provider.backends.ibmq_16_melbourne)
    print(f'Running on {backendOne} and {backendTwo}.\n')
    
else:
    backendOne = provider.backends.ibmq_16_melbourne
    backendTwo = provider.backends.ibmq_16_melbourne
    print(f'Running on {backendOne} and {backendTwo}\n')
    
qc = QuantumCircuit(total)
initialize(qc, [*range(n_qb)])

for i in range(int(np.sqrt(n_qb))):
    qc.append(oracle_maker(n_qb, 1), [0,1,2,6,7,8])
    qc.append(oracle_maker(n_qb, 2), [3,4,5,9,10,11])

    qc.append(diffuser(int(n_qb/2), int(total/2)), [0,1,2,6,7,8])
    qc.append(diffuser(int(n_qb/2), int(total/2)), [3,4,5,9,10,11])

qc.reset([6,7,8,9,10,11])
qc.measure_all()
display(qc.draw(output='mpl',filename='splitgrov110011.png',fold=-1))

jobOne = execute(qc, backend=backendOne, shots=100, optimization_level=3)    
job_monitor(jobOne, interval = 2)
results = job.result()                                               
answer = results.get_counts(qc)                                      
print(answer)
print(f'Time Taken for Job: {results.time_taken}\n')

runtime = results.time_taken
print(f'Total program running time: {runtime}')
display(plot_histogram(answer, figsize=(15,10),color='purple', 
                       title="Combined Results of Split Matrix Grover's"))

######################################
# Additional circuit specifications.
######################################
#Returns the transpiled circuit
trans_circOne = transpile(qcOne, backendOne) 
#Draws the transpiled (larger) circuit.
trans_circOne.draw(filename='trans.png')     
#Returns the size of the transpiled circuit.
sizeOne = trans_circOne.size();      
#Returns the depth of the transpiled circuit.
depthOne = trans_circOne.depth();  
#Prints the transpiled circuit size.
print("Circuit size: %f" % sizeOne)                                
#Prints the transpiled circuit depth.
print("Circuit Depth: %f\n\n" % depthOne)                          
######################################

    


# In[ ]:




