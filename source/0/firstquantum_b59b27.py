# https://github.com/joelshejar/qiskitfirst/blob/655d2bd92640a471e57a45b129dbb003ef3f6d26/firstquantum.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit


# In[2]:


from qiskit import IBMQ


# In[3]:


IBMQ.save_account('e09fd3916c85a0a4c4eb21ffec6ee48758d13a8e3b76c50e6301b7f3e28656d157dd4f406530ae18ccbee21bc6f4685b6271433292b424a81eeb3cebcb04d85b')


# In[4]:


IBMQ.load_account()


# In[5]:


from qiskit import *


# In[6]:


qr = QuantumRegister(2)


# In[7]:


cr = ClassicalRegister(2)


# In[8]:


circuit = QuantumCircuit(qr, cr)


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


circuit.draw()


# In[11]:


circuit.h(qr[0])


# In[12]:


circuit.draw(output='mpl')


# In[13]:


circuit.cx(qr[0], qr[1])


# In[14]:


circuit.draw(output='mpl')


# In[15]:


circuit.measure(qr, cr)


# In[16]:


circuit.draw(output='mpl')


# In[17]:


backend = Aer.get_backend('qasm_simulator')


# In[18]:


job = execute(circuit, backend)


# In[19]:


from qiskit.tools.visualization import plot_histogram


# In[20]:


result = job.result()


# In[21]:


plot_histogram(result.get_counts(circuit))


# In[22]:


provider = IBMQ.get_provider('ibm-q')


# In[24]:


qcomp = provider.get_backend('ibmq_johannesburg')


# In[25]:


qcomp = provider.get_backend('ibmq_16_melbourne')


# In[26]:


IBMQ.load_account()


# In[27]:


provider = IBMQ.get_provider(group='open', project='main')


# In[28]:


backend = provider.get_backend('ibmq_vigo')


# In[29]:


backend.configuration()


# In[30]:


job = execute(circuit, backend)


# In[31]:


from quiskit.tools.monitor import job_monitor


# In[32]:


from qiskit.tools.monitor import job_monitor


# In[33]:


job_monitor(job)


# In[34]:


result = job.result


# In[35]:


plot_histogram(result.get_counts(circuit))


# In[36]:


result = job.result()


# In[37]:


plot_histogram(result.get_counts(circuit))


# In[ ]:




