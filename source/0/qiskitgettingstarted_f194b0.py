# https://github.com/ix-stefstet/qaif/blob/68d981df6193f3b6637676a154ae2f7cef4f4ea4/src/jupyter/archive/Qiskitgettingstarted.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer,
  __qiskit_version__)
from qiskit.visualization import plot_histogram

print(__qiskit_version__)

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)


# In[2]:


circuit.draw()


# In[3]:


plot_histogram(counts)


# In[ ]:




