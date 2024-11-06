# https://github.com/doraparmac/Projekti/blob/aa3980217cb70cd36d5d0a396845738d834eb8b5/Kvantna%20teleportacija/Kvatna_teleportacija2.py
#!/usr/bin/env python
# coding: utf-8

# In[14]:


from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute, Aer
from qiskit import IBMQ
from  qiskit.visualization import plot_histogram
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

circuit = QuantumCircuit(3,3)

get_ipython().run_line_magic('matplotlib', 'inline')


circuit.x(0)
circuit.barrier()


circuit.h(1)
circuit.cx(1,2)

circuit.cx(0,1)
circuit.h(0)
circuit.barrier()

circuit.measure([0,1],[0,1])
circuit.barrier()

circuit.cx(1,2)
circuit.cz(0,2)

circuit.draw(output='mpl')



# In[12]:


circuit.measure(2,2)
simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = simulator, shots = 1024).result()
counts = result.get_counts()
plot_histogram(counts)


# In[ ]:




