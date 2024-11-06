# https://github.com/jonasgutierrez21/learningbasicsofqc/blob/0bca28e3b2f61ee5390fee602fb6b56d81174386/general%20QFT%20circuit,%20with%20n%20qubits.py
#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from numpy import pi

from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools. monitor import job_monitor
from qiskit.visualization import plot_histogram


# In[15]:


def qft_rots(circuit,a):
    if a==0:
      return circuit
    a-=1 # ensures index starts at 0
    circuit.h(a) # hadamard
    for qubit in range (a): # less sign. qubit, smaller angled CROT
       circuit.cp(pi/2**(a-qubit), qubit,a)
    


# In[16]:


qc=QuantumCircuit(4)
qft_rots(qc,4)
qc.draw()


# In[ ]:





# In[ ]:




