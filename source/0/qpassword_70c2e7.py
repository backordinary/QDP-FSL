# https://github.com/Shakhnoza21/ShakhnozaNewbyToQiskit/blob/f9eb2f2914aef8baa3f218e6f20381b0be37409f/QPassword.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from qiskit.tools.visualization import plot_histogram


# In[2]:


number_of_chars = 4


# In[3]:


circuit = QuantumCircuit(number_of_chars,number_of_chars)
circuit.h(range(number_of_chars))
circuit.barrier()
circuit.measure(range(number_of_chars),range(number_of_chars))


# In[4]:


circuit.draw(output='mpl')


# In[5]:


simulator = Aer.get_backend('qasm_simulator')


# In[6]:


result=execute(circuit,backend=simulator).result()


# In[7]:


from qiskit import IBMQ


# In[8]:


IBMQ.load_account()


# In[9]:


provider=IBMQ.get_provider('ibm-q')


# In[10]:


qcomp=provider.get_backend('ibmq_16_melbourne')


# In[11]:


job=execute(circuit, backend= qcomp)


# In[12]:


from qiskit.tools.monitor import job_monitor


# In[13]:


job_monitor(job)


# In[14]:


result=job.result()


# In[15]:


count=result.get_counts()


# In[16]:


print(count)


# In[17]:


from qiskit.tools.visualization import plot_histogram
plot_histogram(count)


# In[ ]:





# In[19]:


type(count)


# In[ ]:





# In[27]:


listInt =[]
password=''
for word in list_keys:
    listInt.append(int(word,2))
    random_character=(int(word,2)+60)
    password += chr(random_character)


# In[28]:


print(password)


# In[ ]:





# In[ ]:





# In[ ]:




