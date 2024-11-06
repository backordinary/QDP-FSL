# https://github.com/CaptainNomad/Quantum-Random-Number-Generator/blob/2983e4206275b47b5b02853b95ce4bc5ab5003b6/Quantum%20Random%20Number%20Generator.py
#!/usr/bin/env python
# coding: utf-8

# # Building a Quantum Random Number Generator (QRNG)
# 
# Multiple methods are used to build QRNG's in industry and today we will build a primitive one using just one Hadamard Gate, i.e, the [`H Gate`](https://qiskit.org/textbook/ch-states/single-qubit-gates.html#hgate) on a simulator. 
# 
# First, let's import what we require from qiskit:

# In[14]:


#initialization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

# importing Qiskit
from qiskit import IBMQ, BasicAer, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

# import basic plot tools
from qiskit.tools.visualization import plot_histogram


# You can ignore the next two lines of code. This will be required when you want to try the QRNG you build on a real device. We'll get to that later. 

# In[15]:


IBMQ.save_account('7c4ac3fe82eea128f7cd6cffd96009a69bb4fe9303c5edc28839dd0de70cf978d1db50d116822e50b10530cdc167cea417bdf4970393540d6d6b07453139ecf7') # you can find your account id on the IBm Quantum Experience website
                                      # in your profile section 


# In[16]:


provider = IBMQ.load_account()


# In[19]:


q2 = QuantumRegister(1)
c2 = ClassicalRegister(1)
qc2 = QuantumCircuit(q2, c2)
qc2.draw()


# Now apply a hadamard gate to put the first qubit into a $|+\rangle$ state. Where, 
# $$ |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$

# In[20]:


qc2.h(0)
qc2.draw()


# We only get results when we measure the qubit and then read that clasical data, which in our case is then store in our classical register `c2`.

# In[21]:


qc2.measure(q2[0], c2[0])
qc2.draw()


# In[24]:


# running and getting results 

backend = Aer.get_backend('qasm_simulator')
job = execute(qc2, backend, shots=100)
backend = provider.get_backend('ibmq_qasm_simulator')
job = execute(qc2, backend=backend, shots=8000, seed_simulator=12345, backend_options={"fusion_enable":True})
result = job.result()
count = result.get_counts()
print(count)


# In[23]:


plot_histogram(count)


# If we re-run the above cell we randomly get the value `0` and `1`. We've created a random number generator!
