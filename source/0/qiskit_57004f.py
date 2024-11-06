# https://github.com/alexb1200/qiskit-testing/blob/972b3deedef48c9ab9bf31168c0c6bb04d6a6aa1/qiskit.py
#!/usr/bin/env python
# coding: utf-8

# In[2]:


#run me first, use shift+enter
get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.extensions import UnitaryGate
import numpy as np
from numpy import pi
from qiskit.tools import outer
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[14]:


#fourier transform, play around, nice visual with the bloch sphere,
#it doesnt use swaps to fix the qubit ordering, basically whats in thhe IBM book
def qftRot(circuit, n,s=0):
    if n ==s:
        return circuit
    
    circuit.h(s)
    
    for qbit in range(n-s-1):
        print(s)
        circuit.cu1(pi/2**(n-qbit-s-1),n-qbit-1,s)
        
    s+=1
    qftRot(circuit,n,s)
qc=QuantumCircuit(3)
qc.x(0)
qc.x(2)
qftRot(qc,3)
qc.draw('mpl')
backend=Aer.get_backend("statevector_simulator")
stateVec=execute(qc,backend=backend).result().get_statevector()
plot_bloch_multivector(stateVec)


# In[12]:


#the prep phase/my algorithm, run this then run qpe  below. If you want to change the input,
#change it here in the variable  input1 and then run qpe again
#use powers of two for p(input=ouput_cleaned)=1,
#the conversion formula is 1-(output/2^counting_bits)=output_cleaned, works for inputs >= 4
input1=4
angle=4*pi/(input1)
cirq=QuantumCircuit(1)
cirq.h(0)
cirq.rx(angle,0)
backend=Aer.get_backend('statevector_simulator')
job=execute(cirq,backend)
result=job.result()
outputState=result.get_statevector(cirq)

cirq=QuantumCircuit(1)
cirq.h(0)
job=execute(cirq,backend)
result2=job.result()
outputState2=result2.get_statevector(cirq)
out=np.outer(outputState,outputState2)

cirq=QuantumCircuit(1)
cirq.x(0)
cirq.h(0)
cirq.rx(angle,0)
backend=Aer.get_backend('statevector_simulator')
job=execute(cirq,backend)
result3=job.result()
outputState3=result3.get_statevector(cirq)

cirq=QuantumCircuit(1)
cirq.x(0)
cirq.h(0)
job=execute(cirq,backend)
result4=job.result()
outputState4=result4.get_statevector(cirq)
out1=np.outer(outputState3,outputState4)
out=out+out1

print(out)
testGate=UnitaryGate(out).control(1)




cirq.draw("mpl")


# In[13]:


#mn is  the number of counting bits, increase means slower preformance higher resolution
#pretty much exactly from the ibm books implmentation copied and messed around with here so that
#I could understand it. 
mn=3

def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(n//2):
        circ.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            circ.cu1(-pi/float(2**(j-m)), m, j)
        circ.h(j)

qpe2 = QuantumCircuit(mn+1, mn)

# Apply H-Gates to counting qubits:
for qubit in range(mn):
    qpe2.h(qubit)

# Prepare our eigenstate |psi>:
qpe2.h(mn)


# Do the controlled-U operations:
#angle =2*pi/30
repetitions = 1
for counting_qubit in range(mn):
    for i in range(repetitions):
        #qpe2.cu1(angle, counting_qubit, mn);
        qpe2.append(testGate,[counting_qubit,mn])
    repetitions *= 2

# Do the inverse QFT:
qft_dagger(qpe2, mn)

# Measure of course!
for n in range(mn):
    qpe2.measure(n,n)

#qpe2.draw(output='mpl')
backend = Aer.get_backend('qasm_simulator')
shots = 2048
results = execute(qpe2, backend=backend, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[ ]:





# In[ ]:




