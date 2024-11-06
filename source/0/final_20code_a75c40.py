# https://github.com/tornadiego/Quantum-Simulation-of-Lennard-Jones-Potential/blob/60efbb03ed912bd6873e6be3e0abc33f473e6b1f/Final%20Code.py
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from iqx import *
from math import *
from qiskit.tools.visualization import plot_histogram
from qiskit.visualization import plot_state_city

# Loading your IBM Q account(s)
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')


# import registers
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

# Defining the system parameters

d_T = 0.05           # time step for each operator (using Suzuki-Trotter's Identity)
T1 = sqrt(2.0)/2     # natural time scale which came after simplication of values of m, r, e.
N = int(T1/d_T)      # number of iterations 
state_list = []      # To save the state after each iteration
v1=0.*d_T            # Potential operator arguments
v2=-0.74024*d_T
v3=-0.65735175*d_T
v4=-0.5803752*d_T

# Initialisation of state
circuit.x(qreg_q[0])

for i in range(0,N):
    # QFT
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[2])
    circuit.cu1(pi/2, qreg_q[1], qreg_q[0])
    circuit.cu1(pi/2, qreg_q[3], qreg_q[2])
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[3])
    # p-operator
    circuit.rz(pi*pi*d_T/4, qreg_q[0])
    circuit.rz(-pi*pi*d_T, qreg_q[1])
    circuit.rz(pi*pi*d_T/4, qreg_q[2])
    circuit.rz(-pi*pi*d_T, qreg_q[3])
    circuit.cp(pi*pi*d_T, qreg_q[0], qreg_q[1])
    circuit.cp(pi*pi*d_T, qreg_q[2], qreg_q[3])
    # QFT^(-1)
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[3])
    circuit.cu1(-pi/2, qreg_q[1], qreg_q[0])
    circuit.cu1(-pi/2, qreg_q[3], qreg_q[2])
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[2])
    # V-operator
    circuit.cx(qreg_q[3], qreg_q[0])
    circuit.cx(qreg_q[3], qreg_q[1])
    circuit.cx(qreg_q[3], qreg_q[2])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.cp(v3, qreg_q[4], qreg_q[1])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.x(qreg_q[0])
    circuit.x(qreg_q[2])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.cp(v3, qreg_q[4], qreg_q[1])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.x(qreg_q[0])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[4])
    circuit.cp(v4, qreg_q[4], qreg_q[0])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[4])
    circuit.x(qreg_q[2])
    circuit.barrier(qreg_q[0], qreg_q[2])
    circuit.x(qreg_q[0])
    circuit.cp(v2, qreg_q[0], qreg_q[2])
    circuit.x(qreg_q[0])
    circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2])
    circuit.x(qreg_q[1])
    circuit.x(qreg_q[2])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[4])
    circuit.cp(v2, qreg_q[4], qreg_q[0])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[4])
    circuit.x(qreg_q[1])
    circuit.x(qreg_q[2])
    circuit.cx(qreg_q[3], qreg_q[0])
    circuit.cx(qreg_q[3], qreg_q[1])
    circuit.cx(qreg_q[3], qreg_q[2])

circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[3])
# Beckend has been chosen for simulating our cicuit
real_device = provider.get_backend('ibmq_vigo')

# Execution of cicuit qc with 8192 shots 
job = execute(circuit,real_device,shots=8192)

# getting information 
count = job.result().get_counts()
print(count)

