# https://github.com/sidejackthenativity/QuantumRandomNumberGenerator/blob/3719063bc4086c57ca14e860baf3364b73cdadc1/RandomNumber.py
#!/usr/bin/env python
# coding: utf-8

# # The following produces a n-bit random number from quantum qbits

# In[101]:


from qiskit import *
from math import *


# Number of bits. Different quantum machines allow for different number of bits

# In[102]:


qnum = 5


# Create classical and quantum registers

# In[103]:


q = QuantumRegister(qnum)
c = ClassicalRegister(qnum)
qc = QuantumCircuit(q, c)


# Create U gates which puts the qbit into a superposition of states (0 or 1) unless measured.

# In[104]:


for x in range(qnum):
    qc.u(pi/2,0,0,q[x])


# In[105]:


get_ipython().run_line_magic('matplotlib', 'inline')
qc.draw(output='mpl')


# In[106]:


for x in range(qnum):
    qc.measure(x,x)
get_ipython().run_line_magic('matplotlib', 'inline')
qc.draw(output='mpl')


# In[107]:


qc.draw()


# In[108]:


simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, backend = simulator, shots = 1).result()
counts = result.get_counts()
mem = result.data()
#from qiskit.tools.visualization import plot_histogram
#plot_histogram(counts)
#print(counts)


# In[109]:


print(mem)


# ## Extract Key as a hex value

# In[110]:


key, value = list(mem['counts'].items())[0]
print(key)


# ## Random number is printed from quantum simulator

# In[111]:


print(int(key,base=16))


# In[112]:


IBMQ.load_account()


# In[113]:


provider = IBMQ.get_provider('ibm-q')


# In[114]:


qcomp = provider.get_backend('ibmq_lima')


# In[115]:



job = execute(qc, backend = qcomp)
from qiskit.tools.monitor import job_monitor
job_monitor(job)
result = job.result()


# In[116]:


counts = result.get_counts()
mem = result.data()
#print(mem)
#print(len(mem['counts']))
key, value = list(mem['counts'].items())[0]

#print(int(key,base=16))


# ## Random number is printed from quantum MACHINE

# In[118]:


print(value)


# In[ ]:




