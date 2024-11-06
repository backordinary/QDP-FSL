# https://github.com/C3lt1c-Viking/QComp_Hello-World/blob/0ee5b967862a24797f440cd6c1ad12945340437d/HelloWorld.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *


# In[2]:


qr = QuantumRegister(2)


# In[3]:


cr = ClassicalRegister(2)


# In[4]:


circuit = QuantumCircuit(qr, cr)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


circuit.draw()


# In[7]:


circuit.h(qr[0])


# In[8]:


circuit.draw(output='mpl')


# In[9]:


circuit.cx(qr[0], qr[1])


# In[10]:


circuit.draw(output='mpl')


# In[11]:


circuit.measure(qr, cr)


# In[12]:


circuit.draw(output='mpl')


# In[13]:


simulator = Aer.get_backend('qasm_simulator')


# In[15]:


result = execute(circuit, backend = simulator).result()


# In[16]:


from qiskit.tools.visualization import plot_histogram


# In[17]:


plot_histogram(result.get_counts(circuit))


# In[18]:


IBMQ.load_account()


# In[19]:


provider = IBMQ.get_provider('ibm-q')


# In[22]:


qcomp = provider.get_backend('ibmq_5_yorktown')


# In[23]:


job = execute(circuit, backend=qcomp)


# In[24]:


from qiskit.tools.monitor import job_monitor


# In[25]:


job_monitor(job)


# In[26]:


result = job.result()


# In[27]:


plot_histogram(result.get_counts(circuit))


# In[ ]:




