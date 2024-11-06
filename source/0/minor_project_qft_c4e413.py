# https://github.com/Archit3115/Minor-Project/blob/d19901d0baa717c032510422293e39e6d2509c10/Minor_Project_QFT.py
#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
from numpy import pi
# importing Qiskit
import qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit.library import QFT
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # Makes the images look nice")


# In[40]:


qc = QuantumCircuit(3)


# In[41]:


qc.h(2)
qc.draw()


# In[42]:


qc.cu1(pi/2, 1, 2) # CROT from qubit 1 to qubit 2
qc.draw()


# In[43]:


qc.cu1(pi/4, 0, 2) # CROT from qubit 2 to qubit 0
qc.draw()


# In[44]:


qc.h(1)
qc.cu1(pi/2, 0, 1) # CROT from qubit 0 to qubit 1
qc.h(0)
qc.draw()


# In[45]:


qc.swap(0,2)
qc.draw()


# In[46]:


def qft_rotations(circuit, n):
    if n == 0: # Exit function if circuit is empty
        return circuit
    n -= 1 # Indexes start from 0
    circuit.h(n) # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        # For each less significant qubit, we need to do a
        # smaller-angled controlled rotation: 
        circuit.cu1(pi/2**(n-qubit), qubit, n)


# In[47]:


qc = QuantumCircuit(4)
qft_rotations(qc,4)
qc.draw()


# In[48]:


def swap_registers(circuit, n):    
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

# Let's see how it looks:
qc = QuantumCircuit(4)
qft(qc,4)
qc.draw()


# In[49]:


# Create the circuit
qc = QuantumCircuit(3)

# Encode the state 5
qc.x(0)
qc.x(2)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # Makes the images fit")
qc.draw()


# In[50]:


backend = Aer.get_backend("statevector_simulator")
statevector = execute(qc, backend=backend).result().get_statevector()
#plot_bloch_multivector(statevector)


# In[51]:


qft(qc,3)
qft = QFT(len(qc))
qc.draw()


# In[22]:


qft.draw()


# In[36]:


qiskit.__qiskit_version__


# In[ ]:




