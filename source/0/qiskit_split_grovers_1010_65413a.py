# https://github.com/xaviermccaig/Quantum-Thesis/blob/dbfb8fc762db2a34230152e7b1b603bcd898ffd0/Qiskit_Split_Grovers_1010.py
#!/usr/bin/env python
# coding: utf-8

# In[47]:


# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator, AerSimulator
from ibm_quantum_widgets import *
from qiskit.tools.monitor import job_monitor
from qiskit.quantum_info.operators import Operator

# General Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from math import sqrt,log
warnings.filterwarnings('ignore') # to filter out dependency warnings

# Loading IBMQ account
provider = IBMQ.load_account()


# In[48]:


#############################################################################
# This function acts as the phase shift component of the Grover iteration.
# A general diffuser, it needs only the number of index and total qubits.
#############################################################################
def diffuser(qb, total):
    qc = QuantumCircuit(total)
    for qubit in range(qb): #from qb
        qc.h(qubit)
    
    for qubit in range(qb):
        qc.x(qubit)
       
    qc.x(total-1)
    qc.h(total-1)
    qc.mct([*range(qb)], total-1)  # multi-controlled Toffoli was total-1 not qb at end
    qc.h(total-1)
    qc.x(total-1)
    
    for qubit in range(qb):
        qc.x(qubit)
    
    for qubit in range(qb):
        qc.h(qubit)
        
    display(qc.draw('mpl'))    # Draws the diffuser
    Diffuser = qc.to_gate()     # Makes the diffuser into an appendable gate
    Diffuser.name = "Diffuser"  # Names the diffuser
    return Diffuser
#############################################################################
# Places all index/key qubits into a superposition state
#############################################################################
def initialize(qc, qubits):
    for q in qubits:
        qc.h(q)    # Applies the Hadamard gate to place all index qubits
                        # into superposition
    return qc
#############################################################################
# Creates the oracles as needed for search. In this case, creates one oracle
# which is run twice, each for `10'.
#############################################################################
def oracle_maker(qb):
    ##10
    qc = QuantumCircuit(qb)
    qc.cx(0,2)
    qc.x(3)
    qc.cx(1,3)
    qc.ccx(2,3,4)
    qc.z(4)
    qc.ccx(2,3,4)
    qc.cx(1,3)
    qc.x(3)
    qc.cx(0,2)
    ##10
    
    display(qc.draw('mpl'))  # Draws the oracle
    oracle = qc.to_gate()     # Turns the oracle into an appendable gate
    oracle.name = "Oracle"    # Names the gate
    return oracle
#############################################################################
# Main section
#############################################################################
n_qb = 2          # number of qubits needed for index values
total = 5         # number of qubits needed in total, including index and ancilla qubits
val = '1010'      # reminder of value to search for
sim = True

if sim is True:
    #backendOne = Aer.get_backend('qasm_simulator')                        # If using local simulator
    #backendTwo = Aer.get_backend('qasm_simulator')                        # If using local simulator
    backendOne = AerSimulator.from_backend(provider.backend.ibmq_athens)   # If simulating real devices
    backendTwo = AerSimulator.from_backend(provider.backend.ibmq_athens)   # If simulating real devices
    print(f'Running on {backendOne}.\n')
else:
    backendOne = provider.backend.ibmq_belem                               # If using real hardware
    backendTwo = provider.backend.ibmq_belem                               # If using real hardware
    print(f'Running on {backendOne} and {backendTwo}\n')
    
indOne = QuantumRegister(n_qb,name='indices')
indTwo = QuantumRegister(n_qb,name='indices')
ancOne = QuantumRegister(total-n_qb,name='ancillae')
ancTwo = QuantumRegister(total-n_qb,name='ancillae')
claOne = ClassicalRegister(n_qb,name='classical')
claTwo = ClassicalRegister(n_qb,name='classical')

qcOne = QuantumCircuit(indOne,ancOne,claOne)            # Creates circuit for the first search, `10'
qcTwo = QuantumCircuit(indTwo,ancTwo,claTwo)            # Creates circuit for the second search, `10'

initialize(qcOne, [*range(n_qb)])                       # Place index qubits into supersposition
initialize(qcTwo, [*range(n_qb)])                       # Place index qubits into supersposition

qcOne.append(oracle_maker(total), [*range(total)])      # Appends the oracle to the first circuit
qcTwo.append(oracle_maker(total), [*range(total)])      # Appends the oracle to the second circuit

qcOne.append(diffuser(n_qb, total), [*range(total)])    # Appends the diffuser to the first circuit
qcTwo.append(diffuser(n_qb, total), [*range(total)])    # Appends the diffuser to the second circuit

qcOne.reset([2,3,4])                                    # Resets the ancilla qubits in first circuit to |0>
qcTwo.reset([2,3,4])                                    # Resets the ancilla qubits in second circuit to |0>

#qcOne.measure_all()
#qcTwo.measure_all()
qcOne.measure(indOne,claOne)                            # Measures the first circuit's indices
qcTwo.measure(indTwo,claTwo)                            # Measures the second circuit's indices
display(qcOne.draw('mpl'))
display(qcTwo.draw('mpl'))


jobOne = execute(qcOne, backend=backendOne, shots=100, optimization_level=3)    
job_monitor(jobOne, interval = 2)
resultsOne = jobOne.result()                                               
answerOne = resultsOne.get_counts(qcOne)                                      
print(answerOne)
display(plot_histogram(answerOne, color='tomato', title="First Run"))       

jobTwo = execute(qcTwo, backend=backendTwo, shots=100, optimization_level=3)    
job_monitor(jobTwo, interval = 2)
resultsTwo = jobTwo.result()                                               
answerTwo = resultsTwo.get_counts(qcTwo)                                      
print(answerTwo)
display(plot_histogram(answerTwo, color='midnightblue', title="Second Run"))       


#runtime = resultsOne.time_taken + resultsTwo.time_taken
#print(f'Total program running time: {runtime}')

######################################
# Additional circuit diagnostics
######################################
trans_circOne = transpile(qcOne, backendOne)                       #Returns the transpiled circuit.
trans_circOne.draw(filename='trans.png')                           #Draws the transpiled (larger) circuit.
sizeOne = trans_circOne.size();                                    #Returns the size of the transpiled circuit.
depthOne = trans_circOne.depth();  
print("Circuit size: %f" % sizeOne)                                #Prints the transpiled circuit size.
print("Circuit Depth: %f\n\n" % depthOne)                          #Prints the transpiled circuit depth.
######################################
######################################
    


# In[ ]:





# In[ ]:




