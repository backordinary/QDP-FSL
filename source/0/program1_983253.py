# https://github.com/akash100302/qiskit_tutorials/blob/59185f4b37b99c64def086fe46bedb0797fd5538/program1.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_version_table', '')


# In[2]:


from qiskit import IBMQ,QuantumCircuit,transpile
from qiskit.tools import job_monitor
from qiskit.visualization import plot_histogram,plot_gate_map


# In[3]:


IBMQ.save_account('31d27cda1fcfab0693ee781c6b72df36d2b059f00442fc36db7965ea7fc3555d18ae52ad7a1ed9493dea1561d5b89486b06f3d2a39b60239d706beedec6e04c5')


# In[ ]:


IBMQ.load_account()
provider=IBMQ.get_provider(hub='ibm-q',group='open',project='main')
backend = provider.get_backend('aer_simulator')


# In[6]:


from qiskit import IBMQ, Aer
provider = IBMQ.load_account()
available_cloud_backends = provider.backends() 
print('\n Cloud backends:')
for i in available_cloud_backends: print(i)

available_local_backends = Aer.backends() 
print('\n Local backends: ')
for i in available_local_backends: print(i)


# In[10]:


qc=QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.measure_all()
qc.draw(output='mpl')


# In[11]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_job_watcher', '')


# In[ ]:


job=backend.run(transpile(qc,backend=backend),shots=1024)


# In[ ]:





# In[ ]:




