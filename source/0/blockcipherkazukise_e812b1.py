# https://github.com/kwdlab/2022-03-se.kazuki/blob/a779f6889c75516738bba082477730a4f6a5ebfd/BlockCipherKazukiSe.py
#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2022 Kazuki Se
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# In[ ]:


from qiskit import IBMQ, BasicAer,Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from numpy import pi
import time


# In[ ]:


def henkan(qc,n,j,l):
    for k in reversed(range(n)):
        if  j>=2**k :
            qc.x(l+k)
            j=j-2**k


# In[ ]:


def XOR(a):
    circuit.cx(qreg_q[a+0], qreg_q[0])
    circuit.cx(qreg_q[a+1], qreg_q[1])
    circuit.cx(qreg_q[a+2], qreg_q[2])
    circuit.cx(qreg_q[a+3], qreg_q[3])


# In[ ]:


def sbox9(qreg_q,circuit):
    circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6], qreg_q[7], qreg_q[8])
    circuit.x(qreg_q[4])
    circuit.x(qreg_q[5])
    circuit.x(qreg_q[6])
    circuit.x(qreg_q[7])
    circuit.cx(qreg_q[0], qreg_q[4])
    circuit.cx(qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[7])
    circuit.cx(qreg_q[3], qreg_q[7])
    circuit.cx(qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[6])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[2], qreg_q[3], qreg_q[6])
    circuit.ccx(qreg_q[2], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[2], qreg_q[8], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[7])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.barrier(qreg_q[1], qreg_q[0], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6], qreg_q[7], qreg_q[8])


# In[ ]:


def sbox8(qreg_q,circuit):
    circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[7], qreg_q[6], qreg_q[5], qreg_q[4])
    circuit.x(qreg_q[4])
    circuit.x(qreg_q[5])
    circuit.x(qreg_q[6])
    circuit.x(qreg_q[7])
    circuit.cx(qreg_q[0], qreg_q[4])
    circuit.cx(qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[7])
    circuit.cx(qreg_q[3], qreg_q[7])
    circuit.cx(qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[6])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[2], qreg_q[3], qreg_q[6])
    circuit.ccx(qreg_q[2], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[4], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[4], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[3], qreg_q[4], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[3], qreg_q[4], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[3], qreg_q[4], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[3], qreg_q[4], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[5], qreg_q[7])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[5], qreg_q[7])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[5], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[5], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[5], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[5], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[4])
    circuit.barrier(qreg_q[7], qreg_q[6], qreg_q[5], qreg_q[4], qreg_q[3], qreg_q[2], qreg_q[1], qreg_q[0])


# In[ ]:


def sbox7(qreg_q,circuit):
    qreg_q = QuantumRegister(7, 'q')
    creg_c = ClassicalRegister(7, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.barrier(qreg_q[6], qreg_q[5], qreg_q[4], qreg_q[3], qreg_q[2], qreg_q[1], qreg_q[0])
    circuit.x(qreg_q[4])
    circuit.x(qreg_q[5])
    circuit.x(qreg_q[6])
    circuit.cx(qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[4])
    circuit.cx(qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[6])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[0], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[0], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[1], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[2], qreg_q[5])
    circuit.cry(pi/4, qreg_q[0], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(-pi/4, qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(pi/4, qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[5])
    circuit.cry(pi/4, qreg_q[0], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[4])
    circuit.cry(pi/4, qreg_q[1], qreg_q[4])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[4])
    circuit.cry(pi/4, qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[6])
    circuit.cry(pi/4, qreg_q[0], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(-pi/4, qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(pi/4, qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[6])
    circuit.x(qreg_q[0])
    circuit.x(qreg_q[1])
    circuit.x(qreg_q[2])
    circuit.x(qreg_q[3])
    circuit.x(qreg_q[4])
    circuit.x(qreg_q[5])
    circuit.x(qreg_q[6])
    circuit.cx(qreg_q[5], qreg_q[0])
    circuit.cx(qreg_q[5], qreg_q[2])
    circuit.cx(qreg_q[5], qreg_q[3])
    circuit.cx(qreg_q[6], qreg_q[1])
    circuit.cx(qreg_q[6], qreg_q[2])
    circuit.cx(qreg_q[6], qreg_q[3])
    circuit.cx(qreg_q[4], qreg_q[0])
    circuit.cx(qreg_q[4], qreg_q[1])
    circuit.cx(qreg_q[4], qreg_q[2])
    circuit.ccx(qreg_q[6], qreg_q[4], qreg_q[0])
    circuit.ccx(qreg_q[6], qreg_q[4], qreg_q[1])
    circuit.ccx(qreg_q[6], qreg_q[4], qreg_q[2])
    circuit.ccx(qreg_q[6], qreg_q[4], qreg_q[3])
    circuit.ccx(qreg_q[6], qreg_q[5], qreg_q[0])
    circuit.ccx(qreg_q[6], qreg_q[5], qreg_q[1])
    circuit.ccx(qreg_q[6], qreg_q[5], qreg_q[2])
    circuit.ccx(qreg_q[6], qreg_q[5], qreg_q[3])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[4])
    circuit.ccx(qreg_q[5], qreg_q[4], qreg_q[3])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.ccx(qreg_q[5], qreg_q[4], qreg_q[3])
    circuit.barrier(qreg_q[6], qreg_q[5], qreg_q[4], qreg_q[3], qreg_q[2], qreg_q[1], qreg_q[0])


# In[ ]:


def sbox9d(qreg_q,circuit):
    circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6], qreg_q[7], qreg_q[8])
    circuit.cx(qreg_q[0], qreg_q[6])
    circuit.x(qreg_q[7])
    circuit.cx(qreg_q[0], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[0], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[6])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[2], qreg_q[8], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[2], qreg_q[8], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.ccx(qreg_q[3], qreg_q[8], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[8])
    circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6], qreg_q[7], qreg_q[8])


# In[ ]:


def sbox8d(qreg_q,circuit):    
    circuit.barrier(qreg_q[7], qreg_q[6], qreg_q[5], qreg_q[4], qreg_q[3], qreg_q[2], qreg_q[1], qreg_q[0])
    circuit.cx(qreg_q[0], qreg_q[6])
    circuit.x(qreg_q[7])
    circuit.cx(qreg_q[0], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[0], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[6])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[3], qreg_q[6], qreg_q[5])
    circuit.ccx(qreg_q[6], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[6])
    circuit.ccx(qreg_q[3], qreg_q[6], qreg_q[5])
    circuit.ccx(qreg_q[6], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[6])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[5])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[6])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[3], qreg_q[6], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[6], qreg_q[7])
    circuit.ccx(qreg_q[2], qreg_q[0], qreg_q[6])
    circuit.ccx(qreg_q[3], qreg_q[6], qreg_q[5])
    circuit.ccx(qreg_q[3], qreg_q[6], qreg_q[7])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[7])
    circuit.ccx(qreg_q[3], qreg_q[7], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[7])
    circuit.ccx(qreg_q[2], qreg_q[7], qreg_q[5])
    circuit.ccx(qreg_q[2], qreg_q[7], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[7])
    circuit.ccx(qreg_q[2], qreg_q[7], qreg_q[5])
    circuit.ccx(qreg_q[2], qreg_q[7], qreg_q[4])
    circuit.barrier(qreg_q[7], qreg_q[6], qreg_q[5], qreg_q[4], qreg_q[3], qreg_q[2], qreg_q[1], qreg_q[0])


# In[ ]:


def sbox7d(qreg_q,circuit):      
    circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6])
    circuit.x(qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[3], qreg_q[6])
    circuit.ccx(qreg_q[1], qreg_q[0], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[6])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[5])
    circuit.ccx(qreg_q[0], qreg_q[3], qreg_q[6])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[4])
    circuit.ccx(qreg_q[1], qreg_q[3], qreg_q[6])
    circuit.cry(pi/4, qreg_q[0], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(-pi/4, qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(pi/4, qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[5])
    circuit.cry(pi/4, qreg_q[0], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(-pi/4, qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(pi/4, qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[6])
    circuit.cry(pi/4, qreg_q[0], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(-pi/4, qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cry(pi/4, qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[5])
    circuit.cry(pi/4, qreg_q[0], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[5])
    circuit.cry(pi/4, qreg_q[0], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[4])
    circuit.cry(pi/4, qreg_q[1], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[5])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[5])
    circuit.cry(pi/4, qreg_q[1], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(-pi/4, qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cry(pi/4, qreg_q[2], qreg_q[6])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cry(-pi/4, qreg_q[3], qreg_q[6])
    circuit.cx(qreg_q[1], qreg_q[3])
    circuit.cry(pi/4, qreg_q[3], qreg_q[6])
    circuit.barrier(qreg_q[6], qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5])
    circuit.cx(qreg_q[4], qreg_q[3])
    circuit.cx(qreg_q[5], qreg_q[1])
    circuit.cx(qreg_q[6], qreg_q[0])
    circuit.cry(pi/4, qreg_q[4], qreg_q[2])
    circuit.cx(qreg_q[4], qreg_q[5])
    circuit.cry(-pi/4, qreg_q[5], qreg_q[2])
    circuit.cx(qreg_q[4], qreg_q[5])
    circuit.cry(pi/4, qreg_q[5], qreg_q[2])
    circuit.cx(qreg_q[5], qreg_q[6])
    circuit.cry(-pi/4, qreg_q[6], qreg_q[2])
    circuit.cx(qreg_q[4], qreg_q[6])
    circuit.cry(pi/4, qreg_q[6], qreg_q[2])
    circuit.cx(qreg_q[5], qreg_q[6])
    circuit.cry(-pi/4, qreg_q[6], qreg_q[2])
    circuit.cx(qreg_q[4], qreg_q[6])
    circuit.cry(pi/4, qreg_q[6], qreg_q[2])
    circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6])


# In[ ]:


#CIPHERA9bit
n=4
key=8
for i in range(2**n):    
    for h in range(2**key):
            qreg_q = QuantumRegister(17, 'q')
            creg_c = ClassicalRegister(17, 'c')
            circuit = QuantumCircuit(qreg_q, creg_c)
            henkan(circuit,n,i,0)
            henkan(circuit,key,h,9)
            XOR(9)
            sbox9(qreg_q,circuit)
            for o in range(n):
                circuit.swap(qreg_q[o], qreg_q[o+4])
            XOR(13)
            for i in range(n):
                circuit.measure(i,i)


# In[ ]:


#CIPHERA8bit
n=4
key=8
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(16, 'q')
        creg_c = ClassicalRegister(16, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,8)
        XOR(8)
        sbox8(qreg_q,circuit)
        for o in range(4):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(12)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERA7bit
n=4
key=8
for i in range(2**n):    
    for h in range(2**key):
            qreg_q = QuantumRegister(15, 'q')
            creg_c = ClassicalRegister(15, 'c')
            circuit = QuantumCircuit(qreg_q, creg_c)
            henkan(circuit,n,i,0)
            henkan(circuit,key,h,7)
            XOR(7)
            sbox7(qreg_q,circuit)
            XOR(11)
            for i in range(n):
                circuit.measure(i,i)


# In[ ]:


#CIPHERB9bit
n=4
key=12
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(21, 'q')
        creg_c = ClassicalRegister(21, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,9)
        XOR(9)
        sbox9(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        for o in range(n+1):
            circuit.reset(qreg_q[o+4])
        XOR(13)
        sbox9(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(17)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERB8bit
n=4
key=12
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(20, 'q')
        creg_c = ClassicalRegister(20, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,8)
        XOR(8)
        sbox8(qreg_q,circuit)
        for o in range(4):
            circuit.swap(qreg_q[o], qreg_q[o+4])
            circuit.reset(qreg_q[o+4])
        XOR(12)
        sbox8(qreg_q,circuit)
        for o in range(4):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(16)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERB7bit
n=4
key=12
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(19, 'q')
        creg_c = ClassicalRegister(19, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,n,h,7)
        XOR(7)
        sbox7(qreg_q,circuit)
        for o in range(n-1):
            circuit.reset(qreg_q[o+4])
        XOR(11)
        sbox7(qreg_q,circuit)
        XOR(15)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERC9bit
n=4
key=16
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(25, 'q')
        creg_c = ClassicalRegister(25, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,9)
        XOR(9)
        sbox9(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        for o in range(n+1):
            circuit.reset(qreg_q[o+4])
        XOR(13)
        sbox9(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        for o in range(n+1):
            circuit.reset(qreg_q[o+4])
        XOR(17)
        sbox9(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(21)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERC8bit
n=4
key=16
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(24, 'q')
        creg_c = ClassicalRegister(24, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,8)
        XOR(8)
        sbox8(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
            circuit.reset(qreg_q[o+4])
        XOR(12)
        sbox8(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
            circuit.reset(qreg_q[o+4])
        XOR(16)
        sbox8(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(20)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERC7bit
n=4
key=16
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(23, 'q')
        creg_c = ClassicalRegister(23, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,n,h,7)
        XOR(7)
        sbox7(qreg_q,circuit)
        for o in range(n-1):
            circuit.reset(qreg_q[o+4])
        XOR(11)
        sbox7(qreg_q,circuit)
        for o in range(n-1):
            circuit.reset(qreg_q[o+4])
        XOR(15)
        sbox7(qreg_q,circuit)
        XOR(19)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERA9bit復号
n=4
key=8
for i in range(2**n):    
    for h in range(2**key):
            qreg_q = QuantumRegister(17, 'q')
            creg_c = ClassicalRegister(17, 'c')
            circuit = QuantumCircuit(qreg_q, creg_c)
            henkan(circuit,n,i,0)
            henkan(circuit,key,h,9)
            XOR(13)
            sbox9d(qreg_q,circuit)
            for o in range(n):
                circuit.swap(qreg_q[o], qreg_q[o+4])
            XOR(9)
            for i in range(n):
                circuit.measure(i,i)


# In[ ]:


#CIPHERA8bit復号
n=4
key=8
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(16, 'q')
        creg_c = ClassicalRegister(16, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,8)
        XOR(12)
        sbox8d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(8)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERA7bit復号
n=4
key=8
for i in range(2**n):    
    for h in range(2**key):
            qreg_q = QuantumRegister(15, 'q')
            creg_c = ClassicalRegister(15, 'c')
            circuit = QuantumCircuit(qreg_q, creg_c)
            henkan(circuit,n,i,0)
            henkan(circuit,key,h,7)
            XOR(11)
            sbox7d(qreg_q,circuit)
            XOR(7)
            for i in range(n):
                circuit.measure(i,i)


# In[ ]:


#CIPHERB9bit復号
n=4
key=12
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(21, 'q')
        creg_c = ClassicalRegister(21, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,9)
        XOR(17)
        sbox9d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        for o in range(n+1):
            circuit.reset(qreg_q[o+4])
        XOR(13)
        sbox9d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(9)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERB8bit復号
n=4
key=12
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(20, 'q')
        creg_c = ClassicalRegister(20, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,8)
        XOR(16)
        sbox8d(qreg_q,circuit)
        for o in range(4):
            circuit.swap(qreg_q[o], qreg_q[o+4])
            circuit.reset(qreg_q[o+4])
        XOR(12)
        sbox8d(qreg_q,circuit)
        for o in range(4):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(8)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERB7bit復号
n=4
key=12
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(19, 'q')
        creg_c = ClassicalRegister(19, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,n,h,7)
        XOR(15)
        sbox7d(qreg_q,circuit)
        for o in range(n-1):
            circuit.reset(qreg_q[o+4])
        XOR(11)
        sbox7d(qreg_q,circuit)
        XOR(7)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERC9bit復号
n=4
key=16
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(25, 'q')
        creg_c = ClassicalRegister(25, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,9)
        XOR(21)
        sbox9d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        for o in range(n+1):
            circuit.reset(qreg_q[o+4])
        XOR(17)
        sbox9d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        for o in range(n+1):
            circuit.reset(qreg_q[o+4])
        XOR(13)
        sbox9d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(9)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERC8bit復号
n=4
key=16
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(24, 'q')
        creg_c = ClassicalRegister(24, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,key,h,8)
        XOR(20)
        sbox8d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
            circuit.reset(qreg_q[o+4])
        XOR(16)
        sbox8d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
            circuit.reset(qreg_q[o+4])
        XOR(12)
        sbox8d(qreg_q,circuit)
        for o in range(n):
            circuit.swap(qreg_q[o], qreg_q[o+4])
        XOR(8)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


#CIPHERC7bit復号
n=4
key=16
for i in range(2**n):    
    for h in range(2**key):
        qreg_q = QuantumRegister(23, 'q')
        creg_c = ClassicalRegister(23, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        henkan(circuit,n,i,0)
        henkan(circuit,n,h,7)
        XOR(19)
        sbox7d(qreg_q,circuit)
        for o in range(n-1):
            circuit.reset(qreg_q[o+4])
        XOR(15)
        sbox7d(qreg_q,circuit)
        for o in range(n-1):
            circuit.reset(qreg_q[o+4])
        XOR(11)
        sbox7d(qreg_q,circuit)
        XOR(7)
        for i in range(n):
            circuit.measure(i,i)


# In[ ]:


sboxt = [
    [0b0000,0b1111],
    [0b0001,0b1110],
    [0b0010,0b1011],
    [0b0011,0b1100],
    [0b0100,0b0110],
    [0b0101,0b1101],
    [0b0110,0b0111],
    [0b0111,0b1000],
    [0b1000,0b0000],
    [0b1001,0b0011],
    [0b1010,0b1001],
    [0b1011,0b1010],
    [0b1100,0b0100],
    [0b1101,0b0010],
    [0b1110,0b0001],
    [0b1111,0b0101],
]


# In[ ]:


sboxtd = [
    [0b1111,0b0000],
    [0b1110,0b0001],
    [0b1011,0b0010],
    [0b1100,0b0011],
    [0b0110,0b0100],
    [0b1101,0b0101],
    [0b0111,0b0110],
    [0b1000,0b0111],
    [0b0000,0b1000],
    [0b0011,0b1001],
    [0b1001,0b1010],
    [0b1010,0b1011],
    [0b0100,0b1100],
    [0b0010,0b1101],
    [0b0001,0b1110],
    [0b0101,0b1111],
]


# In[ ]:


#CIPHERA古典
n=4
for i in range(2**n):
    cleartext=i
    for h in range(2**n):
        key0=h
        for j in range(2**n):
            key1=j
            u=cleartext^key0
            for k in range(2**n):
                if u==sboxt[k][0]:
                    v=sboxt[k][1]
            c=v^key1


# In[ ]:


#CIPHERB古典
n=4
for i in range(2**n):
    cleartext=i
    for h in range(2**n):
        key0=h
        for j in range(2**n):
            key1=j
            for l in range(2**n):
                key2=l
                u=cleartext^key0
                for k in range(2**n):
                    if u==sboxt[k][0]:
                        v=sboxt[k][1]
                w=v^key1
                for k in range(2**n):
                    if w==sboxt[k][0]:
                        x=sboxt[k][1]
                c=x^key2


# In[ ]:


#CIPHERC古典
n=4
for i in range(2**n):
    cleartext=i
    for h in range(2**n):
        key0=h
        for j in range(2**n):
            key1=j
            for l in range(2**n):
                key2=l
                for g in range(2**n):
                    key3=g
                    u=cleartext^key0
                    for k in range(2**n):
                        if u==sboxt[k][0]:
                            v=sboxt[k][1]
                    w=v^key1
                    for k in range(2**n):
                        if w==sboxt[k][0]:
                            x=sboxt[k][1]
                    y=x^key2
                    for k in range(2**n):
                        if y==sboxt[k][0]:
                            z=sboxt[k][1]
                    c=z^key3

