# https://github.com/SPL-LSU/Codes/blob/a28db11b399e6175134e55b973997b67fa44b0df/Pieces%20of%20Circuit/GHZ%204%20qubit%20circuit.py
#!/usr/bin/env python
# coding: utf-8

# In[7]:


import qutip as qt
from qiskit import *
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram
backend = Aer.get_backend('statevector_simulator')


# In[2]:


def state_creation_circuit(index,qubits):
    if index == 1:
        state_create=QuantumCircuit(qubits)
    elif index ==2:
        state_create=QuantumCircuit(qubits)
        state_create.x(0)
    elif index == 3:
        state_create=QuantumCircuit(qubits)
        state_create.x(qubits)
        state_create.x(0)
    elif index == 4:
        state_create=QuantumCircuit(qubits)
        for i in range(qubits):
            state_create.x(i)
    return state_create


# In[3]:


ghz4=state_creation_circuit(1,4)
ghz4.h(0)
ghz4.cx(0,1)
ghz4.cx(1,2)
ghz4.cx(2,3)
ghz4.draw()


# In[8]:


result = execute(ghz4,backend).result()
counts = result.get_counts()
plot_histogram(counts)


# In[ ]:




