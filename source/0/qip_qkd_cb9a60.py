# https://github.com/tylerram93/QKD-Project/blob/b35cdce214357b1de70de297fd80d581fbf7dd25/QIP_QKD.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Tyler Ram
#10-18-2022
#http://www.wthtjsjs.cn/gallery/23-whjj-july-5556.pdf


# In[2]:


#sourced from the first tutorial: https://qiskit.org/documentation/intro_tutorial1.html
import numpy as np
from qiskit.visualization import array_to_latex
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram


# In[2]:


#Coding outline

#STEP 1
#Alice enters a random string of bits 
#We then encode them in either the X or Z basis
#Need to do so randomly or using some quantum operator?

#STEP 2
#Alice sends all the bits to Bob (Eve can either intercept them or not?)
#Two basis allows us to implement the BB84 protocal

#Final step
#Measure the final output from Bob (i.e. the recieved message)


# In[42]:


###########################################
##      Perform three arbitrary rotations
###########################################
q = QuantumRegister(1)

qc = QuantumCircuit(q)
qc.rz(np.pi/2,q)
qc.rx(np.pi/2,q)
qc.rz(np.pi/2,q)

# Measure
qc.measure_all()
qc.draw()


# In[43]:


######################################
##     Use aer simulator
######################################


svsim = Aer.get_backend('aer_simulator')
qobj = assemble(qc)
result = svsim.run(qobj).result()
plot_histogram(result.get_counts())


# In[ ]:





# In[ ]:





# In[ ]:




