# https://github.com/FranSPG/Blog/blob/b2560850180cf4b98e0de5434fbba52f283bf66b/Quantum%20Computing/HadamardGate/hadamardgate.py
#!/usr/bin/env python
# coding: utf-8

# In[20]:


# useful additional packages 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# importing Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit import Aer, IBMQ
import qiskit as qk


# import basic plot tools
from qiskit.tools.visualization import matplotlib_circuit_drawer as circuit_drawer
from qiskit.tools.visualization import plot_histogram, qx_color_scheme


# In[27]:


# Saving and loading my account

IBMQ.save_account(Qconfig.APItoken)
IBMQ.load_accounts()


# In[29]:


# Backends availables
print("Available backends:")
IBMQ.backends()


# In[30]:


from qiskit.backends.ibmq import least_busy

# Choosing the least busy backend
large_enough_devices = IBMQ.backends(filters=lambda x: x.configuration()['n_qubits'] > 3 and
                                                       not x.configuration()['simulator'])
backend = least_busy(large_enough_devices)
print("The best backend is " + backend.name())


# In[31]:


# Creating registers
qr = QuantumRegister(1)
cr = ClassicalRegister(1)


# In[34]:


# Quantum circuit superposition 
qc_superposition = QuantumCircuit(qr, cr)

# Applying the Hadamard gate
qc_superposition.h(qr)
qc_superposition.measure(qr[0], cr[0])


# In[ ]:


# Measuring the qubit 1024 times
job = execute(qc_superposition, backend, shots = 1024)
result = job.result()


# In[67]:


# Plotting the result
plot_histogram(result.get_counts(qc_superposition))

