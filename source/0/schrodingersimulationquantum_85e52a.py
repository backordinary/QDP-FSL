# https://github.com/MayaHL2/SchrodingerSimulationQuantum/blob/fcc50deab476462b117d1e1f4735bcd219e3be88/SchrodingerSimulationQuantum.py
#!/usr/bin/env python
# coding: utf-8

# In[140]:


import matplotlib.pyplot as plt
import numpy as np

# Choosing the potential function 
t = np.arange(0,7,0.01)
U1 = np.sin(np.pi/7*t)
U2 = 1/2*np.sinc(np.pi*t -7)

plt.plot(t,U1)
plt.plot(t,U2)
plt.ylabel('Potential function')

plt.show()


# In[141]:


# Creating a quantum circuit
from qiskit import QuantumCircuit

# Simulator and visualization
from qiskit import Aer
from qiskit.visualization import plot_histogram
from qiskit import assemble
from qiskit.visualization import array_to_latex

# Decomposition of unitary matrix gate into a set of known gates
from qiskit import transpile


# In[142]:


# Mass of the particles 
b = 1/np.sqrt(5)
a = 2/np.sqrt(5)*1j
m = 1j*b/a
print(m)
print(np.absolute(a)**2 + np.absolute(b)**2)
print(np.conj(a)*b+a*np.conj(b))


# In[143]:


S = [[1, 0, 0, 0],
     [0, b, a, 0],
     [0, a, b, 0],
     [0, 0, 0, 1]]

sqc = QuantumCircuit(2,2)
sqc.unitary(S,[0,1])

sqc.draw()


# In[144]:


s = transpile(sqc, basis_gates = ['cx', 'u1', 'u2', 'u3'])
s.draw()


# In[155]:


# Creating the quantum circuit and choosing the number of inputs
# nbr lattice
l = 4
# circuit size
n = (l-1)*2
# iteration
m = 5
qc = QuantumCircuit(n,n)

# Potential function
U = U1

# Distance between the wave function of two points
e = 0.1
initial_qb = 0

# Initial state
initial_state = [0, 1]

qc.initialize(initial_state, initial_qb)

# Quantum gates
for m in range(m):
    for i in range(0,n-1,2):

        if i+2< n:
            qc.cx(i+1,i+2)

        qc.unitary(S,[i,i+1])

        if i == 0:
            qc.p(-e**2*U[i*100], i)
        qc.p(-e**2*U[i*100], i+1)

# Measure
qc.measure(0,0)

for i in range(1, n, 2):
    qc.measure(i,i)

# Drawing the circuit
qc.draw()


# In[156]:


# Creating the simulator
aer_sim = Aer.get_backend('aer_simulator')

qc.save_statevector()
qobj = assemble(qc, shots=8192)
final_state = aer_sim.run(qc).result().get_statevector()


# In[160]:


# Ploting the result as a vector
print(final_state)
array_to_latex(final_state, prefix="\\text{Statevector} = ")


# In[157]:


# Ploting the result as a histogram
hist = job.result().get_counts()
plot_histogram(hist)

