# https://github.com/singhmeet11/QSearch/blob/d178cd89911dc568e122b20e3b636636ba6ac894/QSearch_1.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import qiskit 
from qiskit import *
from qiskit import IBMQ
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from math import *
get_ipython().run_line_magic('matplotlib', 'inline')

qasm_simulator = Aer.get_backend('qasm_simulator')
statevec_sim = Aer.get_backend('statevector_simulator')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

# Loading your IBM Q account(s)
provider = IBMQ.load_account()

import numpy as np
import qiskit 
from qiskit import *
from qiskit import IBMQ
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import execute
from math import *
get_ipython().run_line_magic('matplotlib', 'inline')
import operator

qasm_simulator = Aer.get_backend('qasm_simulator')
statevec_sim = Aer.get_backend('statevector_simulator')

N_LIMIT=8 
M_LIMIT=4


def pattern_matching(total_letters,search_letters,shots):
    #now for the number of qubits refer to the paper
    N = len(total_letters)
    M = len(search_letters)
    
    if N > N_LIMIT:
        raise ValueError("String exceeds limit!")

    if M > M_LIMIT:
        raise ValueError("Search string exceeds limit!")
            
    #bascially these will be the qubits for indexx we will measure them after this
    i_qubits = ceil(log(2*(N-M)))
    t_qubits = i_qubits*M + 1
    ###ancillia qubits
    a_qubits = i_qubits*M
    # index qubits
    s = i_qubits
    t = t_qubits
    #defining the simulator
    simulator = Aer.get_backend('qasm_simulator')
    #quantum circuit 
    circ = QuantumCircuit(t, s)
    # import operator(add the use of operator here)#######################

    
    
    ##now we will be creating the original state
    def input_state(circ):
         
        #index qubits will be in a uniform superposition state
        for a in range(0, s):
            circ.h(a)
        #creating entaglement
        for b in range(0,M-1):
            for c in range(0,s):
                circ.cx(b*s + c,b*s +s +c)
            for c in range(0,s):
                # we will flip the control bit 
                circ.x((b+1)*s - (c+1))
                control_bits = list(range((b+1)*s - (c+1),(b+1)*s))
                #creating entanglement using multiple controlled x gates
                for d in range((b+2)*s-1,s+(b+1)*s - c -1 , -1):
                    circ.mcx(control_bits,d,ancilla_qubits = [d+1])
                #
                circ.x((b+1)*s - (c+1))
                
        return
    # this will be the oracle which is different from 
    def Oracle1(circ,s_base,start_i):
        s = i_qubits
        t_bit = (start_i + 1) * s - 1
        for i, base in enumerate(total_letters):
            if base == s_base:
                # hack to get binary representation of 'i' as a string
                bin_i = format(i, '0'+str(s)+'b')
                # Convert Binary representation 0, 1 to -1, 1 using Phase gate
                for j in range(0, s):
                    if bin_i[j] == '0':
                        circ.x(start_i * s + j)
                ctr_bits = list(range(start_i * s, t_bit))
                # Apply multi controlled CX
                circ.mcx(ctr_bits, t_bit, ancilla_qubits=[t_bit + 1])

                # Uncomputation of Phase gate
                for j in range(0, start_i):
                    if bin_i[j] == '0':
                        circ.x(start_i * s + j)
        return
    # this oracle will do the amplitude modulation (same as gorver's algo)
    def Oracle2(circ):
        final_target = a_qubits - 1
        ############ look into this
        for a in range(0, a_qubits):
            circ.h(a)
            circ.x(a)
        c_bits = list(range(0,final_target))
        circ.mcx(c_bits, final_target , ancilla_qubits = a_qubits)
        
        for a in range(0,a_qubits):
            circ.x(a)
            circ.h(a)
        return
    

    print("Given String: ", total_letters)
    print("Search: ", search_letters)

    input_state(circ)

    grovers_iterations = int(ceil(sqrt(N + M - 1)))
    for a in range(0, grovers_iterations):
        for a in range(0, M):
            Oracle1(circ,search_letters[a], a)
            Oracle2(circ)

    for a in range(0, i_qubits):
        circ.measure(a,a)

    circ.draw(output='mpl')
#     print(circ.qasm())

    #running the circuit on the local simulator
    job = execute(circ, backend = simulator, shots=shots)

    result = job.result()

    counts = result.get_counts(circ)

   

    sorted_counts = dict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True))
 
    
    
    
    #defining a dictonanry for exact position  of the maximum probablity state, i.e 
    
    print('Running on local simulator')
    print('State', '\t\tOccurance')
    
    for quantum_state in sorted_counts:
        print(quantum_state, '\t\t', sorted_counts[quantum_state])
        
    return   


# qpm.xecute()
   # qpm.execute()
    


# In[ ]:




