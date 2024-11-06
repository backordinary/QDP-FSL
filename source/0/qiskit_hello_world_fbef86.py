# https://github.com/mannurulz/qunatumprogramming/blob/1566602e24c6e428883db9cc3b50b9d44fed366d/qiskit_hello_world.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[2]:


circuit = QuantumCircuit(2,2)
circuit.x(0)
circuit.cx(0,1)
circuit.measure([0,1], [0,1])


# In[3]:


circuit.draw()