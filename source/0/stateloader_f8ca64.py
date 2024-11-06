# https://github.com/thesullivantage/rg_quantum_githubCom/blob/917f2d98193e603abda29b7547824f332a033b41/stateLoader.py
#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from qiskit import *
from qiskit.circuit.random import random_circuit
IBMQ.save_account('63660300336241cbc37115ad2a0a31cba961b97ef980b61b34c2da64e2a8d8c27570ad4a3595c9dce0ead7ed3ec149821fd0c4e82da989df70daa33d0e674995')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


def circuit_to_state(circuit):
    return circuit


# In[6]:


qc_raw = QuantumCircuit.from_qasm_file("randomCircuitIdentity.qasm")
qc_raw.draw('mpl')


# In[ ]:




