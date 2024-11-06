# https://github.com/kubatheklamca/pennylane_test/blob/419cd03008355d9644b1ac2dc2e7fbd0c3f1bd05/pennylanetest.py
#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pennylane as qml
from qiskit import IBMQ


# In[2]:


device = qml.device('default.qubit', wires=2)


# In[34]:


@qml.qnode(device)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    return qml.probs(wires=[0,1])


# Wynik zwracany jest w porządku leksykograficznym

# In[22]:


print(circuit())


# # Uruchomienie na backendzie z IBM Q

# # In[33]:


IBMQ.save_account('token', overwrite=True)
quantum_device = qml.device('qiskit.ibmq', wires=2, backend='ibmq_lima')


@qml.qnode(quantum_device)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    return qml.probs(wires=[0,1])
circuit()