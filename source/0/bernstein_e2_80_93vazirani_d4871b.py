# https://github.com/j3nny-zhang/The-Bernstein-Vazirani-algorithm/blob/4d3fdc5be6350218c960fa4ec5dc8598f69767e6/Bernstein%E2%80%93Vazirani.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from qiskit.tools.visualization import plot_histogram


# In[4]:


secretnumber = "100101"


# In[5]:


circuit = QuantumCircuit(len(secretnumber)+1, len(secretnumber))


# In[6]:


circuit.h(range(len(secretnumber)))


# In[7]:


circuit.x(len(secretnumber))


# In[8]:


circuit.h(len(secretnumber))


# In[9]:


circuit.barrier()


# In[10]:


for ii, yesno in enumerate(reversed(secretnumber)):
    if yesno == "1":
        circuit.cx(ii, len(secretnumber))


# In[11]:


circuit.barrier()


# In[12]:


circuit.h(range(len(secretnumber)))


# In[13]:


circuit.barrier()


# In[14]:


circuit.measure(range(len(secretnumber)), range(len(secretnumber)))


# In[15]:


circuit.draw(output = "mpl")


# In[16]:


simulator = Aer.get_backend('qasm_simulator')


# In[17]:


result = execute(circuit, backend = simulator, shots = 1).result()


# In[18]:


counts = result.get_counts()


# In[19]:


print(counts)


# In[ ]:




