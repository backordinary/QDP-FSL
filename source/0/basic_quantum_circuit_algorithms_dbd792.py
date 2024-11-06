# https://github.com/a-menaf-altintas/quantum-computing/blob/037c240c36d52e1240147266d2fdb719f1916283/basic-quantum-circuit-algorithms.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Abdulmenaf Altintas

from qiskit import *
from qiskit.visualization import plot_histogram


# In[2]:


# show figures in line
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


quantum_circuit = QuantumCircuit(3,3)  # use 3 quantum and classical registers


# In[4]:


quantum_circuit.draw(output="mpl") # visuulize initial circuit


# In[5]:


quantum_circuit.h(0)
quantum_circuit.h(1)
quantum_circuit.draw(output="mpl")


# In[6]:


quantum_circuit.cx(1,2)  # 1 for control qubit and 2 is for target qubit
quantum_circuit.measure([0, 1, 2], [0, 1, 2])  # add measurement
quantum_circuit.draw(output="mpl")


# In[7]:


quantum_simulator = Aer.get_backend("qasm_simulator")


# In[8]:


get_result = execute(quantum_circuit, backend=quantum_simulator).result()


# In[9]:


plot_histogram(get_result.get_counts(quantum_circuit))


# In[10]:


# run on real quantum computer

IBMQ.save_account(open("../IBM/useIBMlocally.txt").read())


# In[11]:


IBMQ.load_account()


# In[12]:


qc_providers = IBMQ.get_provider("ibm-q")


# In[13]:


qc_providers.backends()


# In[14]:


for qc_computer in qc_providers.backends():
    try:
        n_qubits = len(qc_computer.properties().qubits)
        print(f"{qc_computer.name()} => {qc_computer.status().pending_jobs} job(s) are pending and {qc_computer.name()} has {n_qubits} qubits.")
    except:
        n_qubits = "quantum_simulator"
        print(f"{qc_computer.name()} is a simulator")
            


# In[15]:


qc_get_computer = qc_providers.get_backend("ibmq_belem")


# In[16]:


# watch job status

import qiskit.tools.jupyter
from qiskit.tools.monitor import job_monitor

get_ipython().run_line_magic('qiskit_job_watcher', '')

job_executed = execute(quantum_circuit, backend=qc_get_computer)

job_monitor(job_executed)


# In[17]:


# get results of computation

qc_compute_result = job_executed.result()

plot_histogram(qc_compute_result.get_counts(quantum_circuit))


# In[ ]:




