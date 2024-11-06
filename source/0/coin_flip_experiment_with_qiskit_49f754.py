# https://github.com/ahkatlio/coin_flip_experiment_with_Qiskit/blob/a72a978415928e08046bebbd4a4024e09bcbd898/coin_flip_experiment_with_Qiskit.py
#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
IBMQ.save_account()
# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()


# In[10]:


qc = QuantumCircuit(2,2)
qc.h(0)
qc.cx(0,1)
qc.measure(range(2),range(2))
qc.draw()


# In[16]:


backend = Aer.get_backend('qasm_simulator')
job_simulator = backend.run(transpile(qc, backend), shots=1024)
result_simulator = job_simulator.result()
counts = result_simulator.get_counts(qc)
print(counts)


# In[17]:


from qiskit.visualization import plot_histogram
plot_histogram(counts)


# In[18]:


provider.backends()


# In[29]:


backend = provider.get_backend('ibmq_qasm_simulator')


# In[ ]:





# In[ ]:





# In[ ]:




