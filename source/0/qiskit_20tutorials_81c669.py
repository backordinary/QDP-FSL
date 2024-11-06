# https://github.com/jakobrichert/Qiskit-projects/blob/f2e8b1f90de90e4bc59ee2f8396a7daece77c307/QIskit%20tutorial%20%231/Qiskit%20tutorials.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *


# In[2]:


cr = ClassicalRegister(2)


# In[3]:


qr = QuantumRegister(2)


# In[4]:


circuit = QuantumCircuit(qr,cr)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


circuit.draw()


# In[7]:


circuit.h(qr[0])


# In[8]:


circuit.draw(output='mpl')


# In[9]:


circuit.cx(qr[0],qr[1])


# In[10]:


circuit.draw()


# In[11]:


circuit.measure(qr, cr)


# In[12]:


circuit.draw(output = 'mpl')


# In[13]:


simulator = Aer.get_backend('qasm_simulator')


# In[14]:


result = execute(circuit, backend = simulator).result()


# In[15]:


from qiskit.tools.visualization import plot_histogram


# In[16]:


plot_histogram(result.get_counts(circuit))


# In[17]:


IBMQ.load_account()


# In[18]:


provider = IBMQ.get_provider('ibm-q')


# In[19]:


qcomp = provider.get_backend('ibmq_bogota')


# In[20]:


job = execute(circuit,backend=qcomp)


# In[21]:


from qiskit.tools.monitor import job_monitor


# In[22]:


job_monitor(job)


# In[23]:


result = job.result()


# In[24]:


plot_histogram(result.get_counts(circuit))







