# https://github.com/a-menaf-altintas/quantum-computing/blob/037c240c36d52e1240147266d2fdb719f1916283/deutsch-jozsa-algorithm.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Abdulmenaf Altintas

# balanced function testing: n bit input


# In[2]:


from qiskit import *
from qiskit.visualization import plot_histogram


# In[3]:


# show figures in line
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# input string
input_string = "1101"


# In[5]:


quantum_circuit = QuantumCircuit(len(input_string) + 1, len(input_string))  # use n quantum and n-1 classical registers


# In[6]:


quantum_circuit.draw(output="mpl") # visualize initial circuit


# In[7]:


# initialize n input register and 1 output register qubits by applying Hadamard gates
for qubit in range(len(input_string)):
    quantum_circuit.h(qubit)
quantum_circuit.x(len(input_string))
quantum_circuit.h(len(input_string))
quantum_circuit.draw(output="mpl")


# In[8]:


# add balanced oracle

quantum_circuit.barrier()

for qubit in range(len(input_string)):
    if input_string[qubit] == "1":
        quantum_circuit.x(qubit)


        
for qubit in range(len(input_string)):
    quantum_circuit.cx(qubit, len(input_string))


    
for qubit in range(len(input_string)):
    if input_string[qubit] == "1":
        quantum_circuit.x(qubit)

quantum_circuit.barrier()

quantum_circuit.draw(output="mpl")


# In[9]:


# apply Hadamard gate to input registers
for qubit in range(len(input_string)):
    quantum_circuit.h(qubit)
quantum_circuit.draw(output="mpl")


# In[10]:


# add measuremnt to input registers
for qubit in range(len(input_string)):
    quantum_circuit.measure(qubit, qubit)

quantum_circuit.draw(output="mpl")


# In[11]:


# simulate on quantum simulator and get results
quantum_simulator = Aer.get_backend("qasm_simulator")
get_result = execute(quantum_circuit, backend=quantum_simulator).result()
plot_histogram(get_result.get_counts(quantum_circuit))


# In[12]:


# run on real quantum computer

IBMQ.save_account(open("../IBM/useIBMlocally.txt").read())


# In[13]:


IBMQ.load_account()


# In[14]:


qc_providers = IBMQ.get_provider("ibm-q")


# In[15]:


qc_providers.backends()


# In[16]:


for qc_computer in qc_providers.backends():
    try:
        n_qubits = len(qc_computer.properties().qubits)
        print(f"{qc_computer.name()} => {qc_computer.status().pending_jobs} job(s) are pending and {qc_computer.name()} has {n_qubits} qubits.")
    except:
        n_qubits = "quantum_simulator"
        print(f"{qc_computer.name()} is a simulator")


# In[17]:


qc_get_computer = qc_providers.get_backend("ibmq_belem")


# In[18]:


# watch job status

import qiskit.tools.jupyter
from qiskit.tools.monitor import job_monitor

get_ipython().run_line_magic('qiskit_job_watcher', '')

job_executed = execute(quantum_circuit, backend=qc_get_computer)

job_monitor(job_executed)


# In[19]:


# get results of computation

qc_compute_result = job_executed.result()

plot_histogram(qc_compute_result.get_counts(quantum_circuit))


# In[ ]:




