# https://github.com/SPL-LSU/Codes/blob/a28db11b399e6175134e55b973997b67fa44b0df/RoyWIP/Circuit%20Basics.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from qiskit import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#quanutm circuit of 3 qubits
circ=QuantumCircuit(3)


# In[3]:


circ.h(0) #hadamard gate on q0
circ.cx(0,1) #cnot gate on control q0 and target q1 (creates a Bell state)
circ.cx(0,2)#this time target is q2 (creates a GHZ state)


# In[4]:


circ.draw()


# In[6]:


#Statevector Backend
#returns quantum state, a vector of 2^n
from qiskit import Aer

backend=Aer.get_backend('statevector_simulator')
#creates a quantum program to execute
job=execute(circ,backend)


# In[10]:


#job object has 2 methods, job.status() and job.result()
status=job.status()
result=job.result()


# In[11]:


#get_statevector returns the final state of a job
outputstate= result.get_statevector(circ, decimals=3)
outputstate


# In[13]:


from qiskit.visualization import plot_state_city
plot_state_city(outputstate)


# In[14]:


#unitary simulator calculates the 2^n x 2^n matrix representing gates 
#only works with unitaries obviously
backend=Aer.get_backend('unitary_simulator')
job=execute(circ, backend)
result = job.result()
print(result.get_unitary(circ, decimals=3))


# In[23]:


import math
circ2=QuantumCircuit(1)
circ2.h(0)
job=execute(circ2, backend)
result=job.result()
print(result.get_unitary(circ2, decimals=3)*math.sqrt(2))
#returns the hadamard matrix


# In[26]:


meas=QuantumCircuit(3,3) #I think the 2nd arguement is 3 classical bits
meas.barrier(range(3)) #prevents multiple gates from 'stacking' (best word I know for it)
#maps q meas to bits
meas.measure(range(3), range(3))

qc=circ+meas
qc.draw()


# In[45]:


from qiskit.tools.visualization import plot_histogram


#returns either a bit string 000 or 111
backend_sim= Aer.get_backend('qasm_simulator')
job_sim=execute(qc, backend_sim, shots=1000)
result_sim=job_sim.result()
counts=result_sim.get_counts(qc)
plot_histogram(counts)


# In[50]:


from qiskit import IBMQ
IBMQ.load_account()
IBMQ.providers()


# In[51]:


#gets a provider from those available
provider= IBMQ.get_provider(group='open')
#list backends
provider.backends()


# In[54]:


result_exp= job_exp.result()
counts_exp = result_exp.get_counts(qc)
plot_histogram([counts_exp,counts], legend=['Device', 'Simulator'])


# In[ ]:




