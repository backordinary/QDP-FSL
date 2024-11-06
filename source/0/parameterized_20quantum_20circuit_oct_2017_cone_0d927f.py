# https://github.com/LilyHeAsamiko/QC/blob/216a52fb15464b238ca8f3903748b745af8f7682/QIML/Parameterized%20Quantum%20Circuit_Oct%2017(Cone).py
#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import *

# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[50]:


#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
# import module for repetition
from qiskit.ignis.verification.topological_codes import RepetitionCode
from qiskit.ignis.verification.topological_codes import lookuptable_decoding
from qiskit.ignis.verification.topological_codes import GraphDecoder
# import basic plot tools
from qiskit.visualization import plot_histogram

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def get_noise(p_meas,p_gate):
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model

# with depth D = 2 and with periodic boundary conditions,there are only two pos-
# sible causal cones: a 4-qubit cone enclosing Nb = 3 blocks
# time step tau = 0.1

def get_raw_results(code,noise_model):#=None
    circuits = code.get_circuit_list()
#    raw_results = {}
#    for log in range(2):
#        job = execute( circuits[log], Aer.get_backend('qasm_simulator'), noise_model=noise_model)
#        raw_results[str(log)] = job.result().get_counts(str(log))
    table_results = {}
    job = execute( circuits[0], Aer.get_backend('qasm_simulator'), noise_model=noise_model, shots=10000 )
    table_results[str(0)] = job.result().get_counts(str(0))
    job = execute( circuits[1], Aer.get_backend('qasm_simulator'), noise_model=noise_model, shots=10000 )
    table_results[str(1)] = job.result().get_counts(str(1))

    P = lookuptable_decoding(raw_results,table_results)
    print('P =',P)
    return [P,table_results]#raw_results

# 1.Cone
tau = 0.1
Nb = 3
Np = Nb/tau
qreg_q = QuantumRegister(4, 'q')
areg_q = QuantumRegister(1,'ancilla_qubit')
creg_c = ClassicalRegister(4, 'c')
qc = QuantumCircuit(qreg_q, areg_q, creg_c)
qc.h(areg_q[0])
qc.cu1(np.pi/Nb, qreg_q[2], 4)
qc.x(qreg_q[2]) 
qc.cx(qreg_q[2],qreg_q[3])
qc.cu1(np.pi/Nb, qreg_q[0], 4)
qc.x(qreg_q[0])
qc.cx(qreg_q[0],qreg_q[1])
qc.cu1(np.pi/Nb, qreg_q[1], 4)
qc.x(qreg_q[1])
qc.cx(qreg_q[1],qreg_q[2])
qc.h(areg_q[0])
qc.measure(qreg_q[0], creg_c[0])
backend = Aer.get_backend('qasm_simulator')
shots = 4096
    #    results = execute(qc, backend=backend, shots=shots).result()
    #    answer = results.get_counts()

    #    n = 4
T = 10

code = RepetitionCode(2,T)

noise_model = get_noise(0.05,0.05)

[P,raw_results] = get_raw_results(code,noise_model)

plt.figure()
plt.bar([0,1],[P['0'],P['1']]) #answer
plt.title(str(0)+'th qubit'+''''s'''+' results')
        
results = code.process_results(raw_results)
#        for log in raw_results:
#            print('Logical',log,':',raw_results[log],'\n')
for log in ['0','1']:
    print('\nLogical ' + log + ':')
    print('raw results       ', {string:raw_results[log][string] for string in raw_results[log] if raw_results[log][string]>=50 })
    print('processed results ', {string:results[log][string] for string in results[log] if results[log][string]>=50 })
    
    for string in results[log]:
        if len(string) >0 & results[log][string]>=50:
            plt.figure()
            plt.bar(results[log][string]) #answer
qc.measure(qreg_q[1], creg_c[1])
backend = Aer.get_backend('qasm_simulator')
shots = 4096
    #    results = execute(qc, backend=backend, shots=shots).result()
    #    answer = results.get_counts()

    #    n = 4
T = 10

code = RepetitionCode(2,T)

noise_model = get_noise(0.05,0.05)

[P,raw_results] = get_raw_results(code,noise_model)

plt.figure()
plt.bar([0,1],[P['0'],P['1']]) #answer
plt.title(str(1)+'th qubit'+''''s'''+' results')
        
results = code.process_results(raw_results)
#        for log in raw_results:
#            print('Logical',log,':',raw_results[log],'\n')
for log in ['0','1']:
    print('\nLogical ' + log + ':')
    print('raw results       ', {string:raw_results[log][string] for string in raw_results[log] if raw_results[log][string]>=50 })
    print('processed results ', {string:results[log][string] for string in results[log] if results[log][string]>=50 })
    
    for string in results[log]:
        if len(string) >0 & results[log][string]>=50:
            plt.figure()
            plt.bar(results[log][string]) #answer
qc.measure(qreg_q[2], creg_c[2])
backend = Aer.get_backend('qasm_simulator')
shots = 4096
    #    results = execute(qc, backend=backend, shots=shots).result()
    #    answer = results.get_counts()

    #    n = 4
T = 10

code = RepetitionCode(2,T)

noise_model = get_noise(0.05,0.05)

[P,raw_results] = get_raw_results(code,noise_model)

plt.figure()
plt.bar([0,1],[P['0'],P['1']]) #answer
plt.title(str(2)+'th qubit'+''''s'''+' results')
        
results = code.process_results(raw_results)
#        for log in raw_results:
#            print('Logical',log,':',raw_results[log],'\n')
for log in ['0','1']:
    print('\nLogical ' + log + ':')
    print('raw results       ', {string:raw_results[log][string] for string in raw_results[log] if raw_results[log][string]>=50 })
    print('processed results ', {string:results[log][string] for string in results[log] if results[log][string]>=50 })
    
    for string in results[log]:
        if len(string) >0 & results[log][string]>=50:
            plt.figure()
            plt.bar(results[log][string]) #answer
qc.measure(qreg_q[3], creg_c[3])
backend = Aer.get_backend('qasm_simulator')
shots = 4096
    #    results = execute(qc, backend=backend, shots=shots).result()
    #    answer = results.get_counts()

    #    n = 4
T = 10

code = RepetitionCode(2,T)

noise_model = get_noise(0.05,0.05)

[P,raw_results] = get_raw_results(code,noise_model)

plt.figure()
plt.bar([0,1],[P['0'],P['1']]) #answer
plt.title(str(3)+'th qubit'+''''s'''+' results')
        
results = code.process_results(raw_results)
#        for log in raw_results:
#            print('Logical',log,':',raw_results[log],'\n')
for log in ['0','1']:
    print('\nLogical ' + log + ':')
    print('raw results       ', {string:raw_results[log][string] for string in raw_results[log] if raw_results[log][string]>=50 })
    print('processed results ', {string:results[log][string] for string in results[log] if results[log][string]>=50 })
    
    for string in results[log]:
        if len(string) >0 & results[log][string]>=50:
            plt.figure()
            plt.bar(results[log][string]) #answer
#P['0'] = [0.1866, 0.2001, 0.1991, 0.1832]
#P['1'] = [0.2176, 0.2164, 0.212, 0.2253]

#plt.figure()
#plt.bar(P) #answer

qc.draw()


    


# In[ ]:




