# https://github.com/damesaa201710054010/QuantumComputing/blob/558f0fa75b7b2a73960d9a7a4022f9d35e965b7e/sourceCode/QuamtunEntangled.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit_textbook.tools import array_to_latex
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[2]:


#Construimos un circuito con 3 registros 
qc = QuantumCircuit(3)
#aplicamos la compuerta H al primer registro
qc.h(0)
#luego aplicamos CNOT en registros 0 y 1, luego 1 y 2
qc.cx(0,1)
qc.cx(1,2)


# In[3]:


#Dibujo del circuito
qc.draw(output='mpl')


# In[4]:


#State vector
backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()
array_to_latex(final_state, pretext="\\text{Statevector = }")


# In[5]:


#Histograma de los resultados
results = execute(qc,backend).result().get_counts()
plot_histogram(results)


# In[6]:




