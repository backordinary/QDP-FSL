# https://github.com/lizzie-m/Quantum-computing/blob/719f9b8ec565eab26e41f58876521fa13845c672/Grovers%20Dinner%20Party%20Problem/Grovers%20dinner%20party%20problem.py
#!/usr/bin/env python
# coding: utf-8

# In[49]:


#Using a quantum simulator to quickly sort out a dinner party problem.
#In the hypothetical situation where Lizzie and Tendai do not like each other
#A probabilty distribution of the possible seating arrangement in alphabetical order can be seen below


from qiskit import BasicAer
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.tools.visualization import plot_histogram


# In[44]:


log_expr = '((Lizzie & Primrose) | (Tino & Tendai)) & ~( Lizzie & Tendai)'
algorithm = Grover(LogicalExpressionOracle(log_expr))


# In[45]:


backend = BasicAer.get_backend('qasm_simulator')


# In[46]:


result = algorithm.run(backend)


# In[47]:


plot_histogram(result['measurement'], title='Possible Party Comnbinations', bar_labels = True)


# In[ ]:




