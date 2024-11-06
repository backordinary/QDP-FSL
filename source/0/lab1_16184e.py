# https://github.com/witseie-elen4022/ELEN4022_LAB1_2022_MANJATE_MUCHAVELELI/blob/de00827e6d33ea29fb00a55a9d03e1eaa187b79c/Lab1.py
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import qiskit
import numpy as np
from qiskit import *
from qiskit.tools.visualization import plot_bloch_multivector
get_ipython().run_line_magic('matplotlib', 'inline')


# In[82]:


wires = 5
#Create a quantum circuit acting on a quantum registry of four cubits
circuit = QuantumCircuit(wires)

#Add a Cx(CNOT) gate on control qubit 0 and target qubit 1 
k = wires - 1
for x in range(k):
    for y in range(k):
        circuit.cx(x,y+1+x)
    k = k-1

circuit.draw('mpl')


# In[84]:


wires = 10
#Create a quantum circuit acting on a quantum registry of four cubits
circuit = QuantumCircuit(wires)

#Add a Cx(CNOT) gate on control qubit 0 and target qubit 1 
k = wires - 1
for x in range(k):
    for y in range(k):
        circuit.cx(x,y+1+x)
    k = k-1

circuit.draw('mpl')


# In[83]:


#create an identity matrix
I = np.identity(2)
cnot = np.identity(4)
cnot[[1,3]] = cnot[[3,1]]
mat = np.zeros([4, 4])
X = np.array([[0, 1], [1, 0]])
ss = np.array([[0, 1]])
CC = np.kron(X,X)
II = np.identity(2)
C = np.kron(CC,cnot)
CX = np.roll(C,1,axis=0)
print(C)



# In[54]:


#Import Aer
from qiskit import Aer


# In[55]:


#Run the quantum circuit on a unitary simulator
backend = Aer.get_backend('unitary_simulator')
job = execute(circuit, backend, shots=1024)
result = job.result()
unitaryvector = result.get_unitary(circuit, decimals=3)

#Show results
plot_bloch_multivector(unitaryvector)


# In[56]:


print(unitaryvector)


# In[15]:


#Import standard Python libraries
import numpy as np
import math

#A suggested libay for the linear algebra task
from functools import reduce

#Forpurposes of visualistions in this specific lab
np.set_printoptions(precision=2, suppress=True)

#Complex numbers
w = 2.3 + 5.1j
print(f'w = {w}')
print(f'|2.3+5.1j| = {np.round(abs(w), 2)}')

r = abs(w)
phi = np.arctan2(w.imag, w.real)
z = r*np.exp(1j*phi)


# In[16]:


H = 1/math.sqrt(2)*np.array([[1, 1], [1, -1]])
print(f'HH† = |H><H| = \n{H @ H.T.conj()}\n')

H2 = np.kron(H, H)
print(f'HkronH = \n{H2}\n')

e1 = np.array([1, 0])
ket00 = np.kron(e1, e1)
print(f'H|00> = \n{H2@ket00}\n')


# In[17]:


#Statevector backend

#Import Aer
from qiskit import Aer

#Run the quantum circuit on the statevector backend
backend = Aer.get_backend('statevector_simulator')

#Create a quantum program for execution
job = execute(circuit, backend)
result = job.result()
outputstate = result.get_statevector(circuit, decimals=3)
print(outputstate)


# In[ ]:




