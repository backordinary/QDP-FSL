# https://github.com/licTomasPerez/Quantum-Computing-Qiskit/blob/5669e1740189820c3a5f02b2692f6774fc576672/QC%20with%20Qiskit.py
#!/usr/bin/env python
# coding: utf-8

# In[3]:


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
q = QuantumRegister(1) 
c = ClassicalRegister(1)

qc = QuantumCircuit(q,c)


# In[4]:


qc.draw('mpl')


# In[6]:


from qiskit.quantum_info import Statevector


# In[8]:


psi1=Statevector(qc)
psi1.draw('latex')


# In[10]:


from qiskit.visualization import plot_state_qsphere
plot_state_qsphere(qc)

## Bloch sphere with a color map which represents the relative phase of 
## said phase and a cycle that represents the probability of measuring that particular state.
## The relative phase of state psi1=\ket{0} is 0 and the angle indicates a 100% probability of 
## measuring \ket{0}


# In[16]:


## The density matrix for this density matrix can be computed as follow

from qiskit.quantum_info import DensityMatrix
rho1=DensityMatrix(qc)
rho1.draw('latex', prefix='\\rho_1 =') 


# In[17]:


## Visualizing said density matrix

from qiskit.visualization import plot_state_city
plot_state_city(qc)

## Hinton-like diagram for said density matrix


# In[20]:


## Looking the previous diagram from above for \ket{0}\bra{0}

from qiskit.visualization import plot_state_hinton
plot_state_hinton(qc)


# In[21]:


## Let's write \ket{0} as a linear combination of Pauli's gates. 

from qiskit.visualization import plot_state_paulivec
plot_state_paulivec(qc)


## This diagram tell's us that \rho_1 can be written as \rho_1 = \frac{1}{2}(\mathds{1}_2 + \sigma_x)


# In[22]:


from qiskit.visualization import plot_bloch_multivector

plot_bloch_multivector(qc)

## rho_1 is represented by a length-one vector pointing to the north pole of the Bloch Sphere


# In[23]:


## Purity of said state

purity = rho1.purity()
print(purity)

## This state is pure thus, purity is equal to one.


# In[ ]:




