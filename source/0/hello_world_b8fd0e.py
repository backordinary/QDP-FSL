# https://github.com/tinegachris/Quantum-Computing/blob/f3214cb5b9da8e8fc461eaa04b1e502b4f2dc43a/Hello_World.py
#!/usr/bin/env python
# coding: utf-8


#Hello_Quantum


# In[1]:


from qiskit import *


# In[2]:


qr = QuantumRegister (2)


# In[3]:


cr = ClassicalRegister (2)


# In[4]:


circuit = QuantumCircuit (qr,cr)


# In[5]:


circuit.draw ()


# In[6]:


circuit.h (qr[0])


# In[8]:


circuit.draw (output='mpl')


# In[9]:


circuit.cx (qr[0],qr[1])


# In[10]:


circuit.draw (output='mpl')


# In[11]:


circuit.measure (qr,cr)


# In[12]:


circuit.draw (output='mpl')


# In[13]:


simulator = Aer.get_backend ('qasm_simulator')


# In[15]:


result = execute(circuit, backend=simulator).result


# In[17]:


from qiskit.tools.visualization import plot_histogram


# In[21]:


plot_histogram (result.get_counts(circuit))


# In[23]:


IBMQ.load_account()


# In[25]:


provider = IBMQ.get_provider ('ibm-q')


# In[26]:


qcomp = provider.get_backend ('ibmq_16_melbourne')


# In[28]:


job = execute(circuit, backend=qcomp)


# In[29]:


from qiskit.tools.monitor import job_monitor 


# In[30]:


job_monitor (job)


# In[31]:


result = job.result ()


# In[32]:


plot_histogram (result.get_counts(circuit))


# In[ ]:




