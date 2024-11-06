# https://github.com/B10-H4ck3r/QComp_SecretBox/blob/943bb63abc2ba7070fd920e1606adbc8b5258377/SecretBox.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from qiskit.tools.visualization import plot_histogram


# In[90]:


secretnumber = '10'


# In[91]:


circuit = QuantumCircuit(len(secretnumber)+1, len(secretnumber))

#circuit.h([0,1,2,3,4,5])
circuit.h(range(len(secretnumber)))
circuit.x(len(secretnumber))
circuit.h(len(secretnumber))

circuit.barrier()

for ii, yesno in enumerate(reversed(secretnumber)):
    if yesno == '1':
        circuit.cx(ii, len(secretnumber))

#circuit.cx(5,6)
#circuit.cx(3,6)
#circuit.cx(0,6)

circuit.barrier()
#circuit.h([0,1,2,3,4,5])
circuit.h(range(len(secretnumber)))

circuit.barrier()

circuit.measure(range(len(secretnumber)), range(len(secretnumber)))


# In[92]:


circuit.draw(output='mpl')


# In[93]:


simulator = Aer.get_backend('qasm_simulator')


# In[94]:


result = execute(circuit, backend = simulator, shots = 1).result()
counts = result.get_counts()
print(counts)


# In[ ]:




