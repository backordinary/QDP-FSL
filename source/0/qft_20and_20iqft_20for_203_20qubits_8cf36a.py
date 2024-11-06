# https://github.com/makotonakai/Quantum-Fourier-transform/blob/01ec55dabad757ce7d67f74ae3154da80fa502c0/QFT%20and%20iQFT%20for%203%20qubits.py

# coding: utf-8

# In[9]:


from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
import numpy as np

APItoken = "Replace me"
url = "Replace me"
IBMQ.enable_account(APItoken, url)

IBMQ.backends()

num_of_qubits = 3
q = QuantumRegister(num_of_qubits)
c = ClassicalRegister(num_of_qubits)
qc = QuantumCircuit(q, c)

#Quantum Fourier Transform
qc.h(q[0])
qc.cu1(np.pi/2,q[0],q[1])
qc.cu1(np.pi/4,q[0],q[2])
qc.h(q[1])
qc.cu1(np.pi/2,q[1],q[2])
qc.h(q[2])

#Inverse Quantum Fourier Transform
#qc.h(q[0])
#qc.cu1(-np.pi/2,q[0],q[1])
#qc.cu1(-np.pi/4,q[0],q[2])
#qc.h(q[1])
#qc.cu1(-np.pi/2,q[1],q[2])
#qc.h(q[2])
for i in range(num_of_qubits):
	qc.measure(q[num_of_qubits-i-1],c[i])

#Put a real device first and a simulator next.	
backends = ['ibmq_20_tokyo', 'qasm_simulator']

#Use this for the actual machine
backend_sim = IBMQ.get_backend(backends[0])
#QFT{'101': 388, '110': 379, '000': 680, '111': 504, '010': 532, '011': 606, '001': 486, '100': 521}
#QFT&iQFT{'101': 198, '110': 140, '000': 855, '111': 210, '010': 871, '011': 850, '001': 726, '100': 246}

#Use this for the simulation
#backend_sim = Aer.get_backend(backends[1])
#QFT{'101': 475, '110': 532, '000': 545, '111': 514, '010': 478, '011': 521, '001': 492, '100': 539}
#QFT&iQFT{'000': 4096}

result = execute(qc, backend_sim, shots=4096).result()

#You can get the quantum circuit drawn in Latex style
#circuit_drawer(qc).show()

print(result.get_counts(qc))
plot_histogram(result.get_counts(qc))



plot_histogram(result.get_counts(qc))

