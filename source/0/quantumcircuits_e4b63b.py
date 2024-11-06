# https://github.com/B10-H4ck3r/QComp_QuantumCircuits/blob/cbb1a0b8bad7a02697beddd9eabc6397ddf9a389/QuantumCircuits.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *


# In[2]:


from qiskit.tools.visualization import plot_bloch_multivector


# In[7]:


circuit = QuantumCircuit(1,1)
circuit.x(0)
simulator = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend = simulator).result()
statevector = result.get_statevector()
print(statevector)
get_ipython().run_line_magic('matplotlib', 'inline')
circuit.draw(output='mpl')


# In[8]:


plot_bloch_multivector(statevector)


# In[10]:


circuit.measure([0],[0])
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = backend, shots = 1024).result()
counts = result.get_counts()
from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)


# In[13]:


circuit = QuantumCircuit(1,1)
circuit.x(0)
simulator = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend = simulator).result()
unitary = result.get_unitary()
print(unitary)


# In[ ]:




