# https://github.com/jonasgutierrez21/learningbasicsofqc/blob/0bca28e3b2f61ee5390fee602fb6b56d81174386/QFT%20example.py
#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
from numpy import pi

from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram


# In[38]:


qc=QuantumCircuit(3)


# In[39]:


#we are going to apply Hadamard gates, to qubit 2
qc.h(2)
qc.draw()


# In[40]:


qc.cp(pi/2,1,2) #  CROT, from qubit 1 to 2, we are turning it an extra 1/4
qc.draw()


# In[41]:


qc.cp(pi/4,0,2) # CROT from qubit  2 to 0, we are turning it another 1/8
qc.draw()


# In[42]:


qc.h(1)
qc.cp(pi/2,0,1) # CROT from qubit 1 to 0
qc.h(0)
qc.draw()


# In[43]:


# we are now going to swap  the qubits to complete the QFT
qc.swap(0,2)
qc.draw()


# In[ ]:





# In[ ]:




