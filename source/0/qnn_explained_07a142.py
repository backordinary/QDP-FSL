# https://github.com/j-a-whl/QNN-proj-auc/blob/6120c388fa306ed58b50b4e18964e4443016dc2f/old/QNN_explained.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import OpflowQNN


# In[2]:


# set method to calculcate expected values
expval = AerPauliExpectation()

# define gradient method
gradient = Gradient()

# define quantum instances (statevector and sample based)
qi_sv = QuantumInstance(Aer.get_backend('statevector_simulator'))

# we set shots to 10 as this will determine the number of samples later on.
qi_qasm = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=10)


# In[3]:


#We first construct the quantum circuit, which is a part of the neural network (see https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html)
params1 = [Parameter('input1'), Parameter('weight1')] # specifies the two variables on which the neural network depends - the input and the weights
qc1 = QuantumCircuit(1) #defines a quantum circuit with one qubit
qc1.h(0) #applies Haddamar to the qubit
qc1.ry(params1[0], 0) #applies a rotation gate to the qubit, rotating by the angle equal to the first input parameter
qc1.rx(params1[1], 0) #-\\- angle equal to second input parameter
qc_sfn1 = StateFn(qc1) #combines the circuit into an object we can play with

# construct cost operator
H1 = StateFn(PauliSumOp.from_list([('Z', 1.0), ('X', 1.0)]))
#when we apply this cost operator to a circuit, it spits out the loss function

#so we find the loss function by combing the operator and circuit as follows:
op1 = ~H1 @ qc_sfn1

print(op1)


# In[4]:


#Now, we bring this all together to a nerual network as follows:
# construct OpflowQNN with the operator, the input parameters, the weight parameters, 
# the expected value, gradient, and quantum instance.
qnn1 = OpflowQNN(op1, [params1[0]], [params1[1]], expval, gradient, qi_sv)
#such that the only input of our neural network is the parameters


# In[5]:



# define (random) input and weights
input1 = np.random.rand(qnn1.num_inputs)
weights1 = np.random.rand(qnn1.num_weights)


# In[6]:


#now if we want to train the network on a single input, so predict the output given guess weights, compute the loss function, and try to minimise it using gradient descend, we do:\
# QNN backward pass
qnn1.backward(input1, weights1)


# In[7]:


#Alternatively, we can do this with all the data points we have in the trainin data
# QNN batched backward pass
qnn1.backward([input1, input1], weights1)


# In[ ]:




