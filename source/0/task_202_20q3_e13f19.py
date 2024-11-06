# https://github.com/Ibituyi/Screening-Task2/blob/72f3f43099864cec4c3b73ffaec4a6eba7e038f7/Task%202,%20Q3.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import QuantumRegister 
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute,IBMQ
from qiskit.tools.monitor import job_monitor

print('\nSign Flip Code')
print('----------------')

IBMQ.enable_account('7a14c59cb5d061ddf7b9287874a80aa122f1cae65910e33fa678220ec5d475a637b83a0e9f8f54a43e3a128d5e5154f084a924dbfdef0da5a3e777c940a15b55')
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator')

# Creating Qubits
q = QuantumRegister(3)
# Creating qubits
circuit = QuantumCircuit(q)
# Hadamard Gate on the first Qubit
circuit.h(q[0])
# Pauli Z Gate on the the first qubit 
circuit.z(0)
# Pauli Z Gate on the the second qubit
circuit.z(1)
# CNOT Gate on the first and second Qubits
circuit.cx(q[0], q[1])
# CNOT Gate on the first and third Qubits
circuit.cx(q[0],q[2])

circuit.h(q[0])
circuit.h(q[1])
circuit.h(q[2]) 
circuit.z(q[0]) #Add this to simulate a sign flip error
circuit.h(q[0])
circuit.h(q[1])
circuit.h(q[2])

circuit.cx(q[0],q[1])
circuit.cx(q[0],q[2])
circuit.ccx(q[2],q[1],q[0])


# In[ ]:




