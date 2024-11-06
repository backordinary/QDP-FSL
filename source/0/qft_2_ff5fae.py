# https://github.com/SPL-LSU/Codes/blob/a28db11b399e6175134e55b973997b67fa44b0df/RoyWIP/QFT(2).py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roy Pace
#1/27/2020
#This code produces a Quantum Fourier Transform for m number of qubits


# In[2]:


#Papers Looked At:
#https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html#example1
#https://arxiv.org/pdf/1903.04359.pdf
#https://quantum-computing.ibm.com/support
#https://qiskit.org/documentation/api/qiskit.extensions.standard.Cu1Gate.html?highlight=cu1
#http://pages.cs.wisc.edu/~dieter/Papers/vangael-thesis.pdf
#https://arxiv.org/pdf/1804.03719.pdf


# In[3]:


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
import math as math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Create the Circuit
m= 5#number of qubits, must be greater than 0
q=QuantumRegister(m)
c=ClassicalRegister(m)
qc= QuantumCircuit(q,c)


# In[5]:


#applies the hadamard 
for i in range(0,m):
    qc.h(q[i])
    #applies the appropriate rotation gates
    for j in range(i+1,m):
            qc.cu1(math.pi/(2**(j)),q[j],q[i])


# In[6]:


qc_i=qc.inverse()
qc.append(qc_i, qc.qubits[:m])


# In[7]:


#Swap gate is not needed            
#for i in range(0,int(m/2)):
    #qc.swap(i,m-i-1)
qc.measure(q,c)
qc.draw()


# In[8]:


#Execute the Circuit
from qiskit.circuit import Gate
from qiskit import execute, Aer

shots = 20000
job = execute(qc, Aer.get_backend('qasm_simulator'), shots=shots)
counts = [job.result().get_counts(i) for i in range(len(job.result().results))]
from qiskit.visualization import plot_histogram
plot_histogram(counts)


# In[9]:


from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')

shots = 1024

provider = IBMQ.get_provider(group='open')

backend = provider.get_backend('ibmq_16_melbourne')
i=1
sum = []
counts = []
avg = 0
for i in range(1,13):
    
    exp_job = execute(qc, backend)
    job_monitor(exp_job)
    exp_counts = exp_job.result().get_counts()
    counts.append(exp_counts)
    sum.append(exp_counts["00000"]/shots)
    i+=1
#avg = sum/(i-1)
print(sum)
#plot_histogram(exp_counts)


# In[10]:


counts


# In[12]:


sum


# In[ ]:


from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')


provider = IBMQ.get_provider(group='open')

backend = provider.get_backend('ibmq_16_melbourne')
exp_job = execute(qc, backend)
job_monitor(exp_job)
exp_counts = exp_job.result().get_counts()
plot_histogram(exp_counts)


# In[ ]:


#Fidelity Calculations
exp_counts


# In[16]:


avg=0
i=0
for i in range(0,12):
    avg+=sum[i]
    print(i)
avg=avg/(i+1)
print(avg)


# In[ ]:


exp_counts["0 0000"]/shots


# In[ ]:


answer["0 0000"]/shots


# In[ ]:


from qiskit.quantum_info import state_fidelity
state_fidelity(answer,exp_counts)


# In[ ]:


state_fidelity()

