# https://github.com/robinpx/quantum-algorithms/blob/85543706c355fe7ebfb55694600d19274f924794/code/.ipynb_checkpoints/random_circ_final-checkpoint.py
#!/usr/bin/env python
# coding: utf-8

# In[60]:


from qiskit.circuit.random import random_circuit


# In[61]:


qubits = 5
circuit_depth = 2
my_circ = random_circuit(qubits, circuit_depth, max_operands=3,measure=True)
my_circ.draw(output='text')


# In[62]:


my_circ2 = my_circ.copy()
my_circ2.draw(output='mpl')
qasm_str = my_circ2.qasm()
print(qasm_str)


# In[64]:


# parse qasm string
qasm_dict = qasm_str.split('\n')
qasm_dict[5] = qasm_dict[5].replace("4.626865", "101010")
qasm_str2 = '\n'.join(qasm_dict)
print(qasm_str2)


# In[65]:


my_circ2=my_circ2.from_qasm_str(qasm_str2)


# In[70]:


my_circ.draw(output='mpl')


# In[71]:


my_circ2.draw(output='mpl')


# In[122]:


import numpy as np
from qiskit import QuantumCircuit

# Create a Quantum Circuit acting on a quantum register of three qubits
circ = QuantumCircuit(3)

# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)

circ.draw("mpl")

# Create a Quantum Circuit
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
# map the quantum measurement to the classical bits
meas.measure(range(3), range(3))

# The Qiskit circuit object supports composition.
# Here the meas has to be first and front=True (putting it before)
# as compose must put a smaller circuit into a larger one.
qc = meas.compose(circ, range(3), front=True)


# In[123]:


# Create a Quantum Circuit
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
# map the quantum measurement to the classical bits
meas.measure(range(3), range(3))

# The Qiskit circuit object supports composition.
# Here the meas has to be first and front=True (putting it before)
# as compose must put a smaller circuit into a larger one.
qc = meas.compose(circ, range(3), front=True)


# In[124]:



# Adding the transpiler to reduce the circuit to QASM instructions
# supported by the backend
from qiskit import transpile

# Use Aer's qasm_simulator
from qiskit.providers.aer import QasmSimulator

backend = QasmSimulator()
shots=1024

# First we have to transpile the quantum circuit
# to the low-level QASM instructions used by the
# backend
qc_compiled = transpile(qc, backend) # edit here replace qc with circuit

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = backend.run(qc_compiled, shots=shots)

# Grab the results from the job.
result_sim = job_sim.result()

counts = result_sim.get_counts(qc_compiled)
print(counts)


# In[125]:


from qiskit.visualization import plot_histogram
plot_histogram(counts)


# In[126]:


# probabilities
for k,v in counts.items():
    print("probability of ", k," is ", v/shots)


# In[127]:


# entropy

# calculate the entropy for a dice roll
from math import log
# the number of events
n = shots
# probability of each event 
for k,v in counts.items():
    p = v/n
    print("probability of ", k," is ",p)
    # calculate entropy
    entropy = -sum([p * log(p) for _ in range(n)])
    # print the result
    print('entropy: %.3f nats' % entropy)


# In[ ]:




