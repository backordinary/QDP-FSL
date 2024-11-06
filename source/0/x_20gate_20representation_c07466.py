# https://github.com/jakobrichert/Qiskit-projects/blob/f2e8b1f90de90e4bc59ee2f8396a7daece77c307/X%20gate%20Representation/X%20gate%20representation.py
#!/usr/bin/env python
# coding: utf-8
# In[1]:


from qiskit import *


# In[3]:


from qiskit.tools.visualization import plot_bloch_multivector


# In[4]:


circuit = QuantumCircuit(1,1)


# In[5]:


circuit.x(0)


# In[7]:


simulator = Aer.get_backend('statevector_simulator')


# In[8]:


result = execute(circuit, backend = simulator).result()


# In[9]:


statevector = result.get_statevector()


# In[10]:


print(statevector)


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


circuit.draw(output='mpl')


# In[17]:


plot_bloch_multivector(statevector)


# In[18]:


circuit.measure([0],[0])


# In[19]:


backend = Aer.get_backend('qasm_simulator')


# In[20]:


result = execute(circuit, backend = backend, shot = 1024).result()


# In[21]:


counts = result.get_counts()


# In[23]:


from qiskit.tools.visualization import plot_histogram


# In[24]:


plot_histogram(counts)


# In[27]:


circuit = QuantumCircuit(1,1)
circuit.x(0)
simulator = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend = simulator).result()
unitary = result.get_unitary()
print(unitary)


# In[ ]:




