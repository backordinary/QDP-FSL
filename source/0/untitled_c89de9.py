# https://github.com/josephkatabira/Test/blob/36e93505c1f89a8254f0aec00932f11e8dfdcc7f/Untitled.py
#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
import matplotlib.pyplot as plt

# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()


# In[21]:


# line for classical 
x1 = [0,8,16,32,64]
y1 = [0,8,16,32,64]
# plotting the line 1 points 
plt.plot(x1, y1, label = "classical", color='green', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize= 5)
    
# line for quantum
x2 = [0,8,16,32,64]
y2 = [0,3,4,6,8]
# plotting the line 2 points 
plt.plot(x2, y2, label = "quantum", color='orange', linewidth = 3,
         marker='o', markerfacecolor='black', markersize= 5)
  
# naming the x axis
plt.xlabel('search space size')
# naming the y axis
plt.ylabel('number of iterations')
# giving a title to my graph
plt.title('comparison of computational complexity')
  
# show a legend on the plot
plt.legend()
  


# In[ ]:




