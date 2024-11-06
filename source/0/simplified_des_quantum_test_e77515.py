# https://github.com/kwdlab/2022-03-matsuura.tatsuya/blob/e4ff5164214b2bc8fdd0ee2ea21fff4db069d9fd/Simplified_DES_Quantum_test.py
#!/usr/bin/env python
# coding: utf-8

"""
Copyright (c) 2021 Tatsuya Matsuura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numpy import pi
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.visualization import plot_histogram



#CCCNOT, CCCCNOT, P10, P8, LS1, LS2, LS-2, IP, IP-1, E/P, S0, S1の量子回路を事前にプログラム
c3x = QuantumCircuit(4, name = 'CCCNOT')
c3x.cry(pi/4, 0 ,3)
c3x.cx(0, 1)
c3x.cry(-pi/4, 1, 3)
c3x.cx(0, 1)
c3x.cry(pi/4, 1, 3)
c3x.cx(1, 2)
c3x.cry(-pi/4, 2, 3)
c3x.cx(0, 2)
c3x.cry(pi/4, 2, 3)
c3x.cx(1, 2)
c3x.cry(-pi/4, 2, 3)
c3x.cx(0, 2)
c3x.cry(pi/4, 2, 3)
c3x.draw(output = 'mpl')
cccnot = c3x.to_gate()

c4x = QuantumCircuit(5, name = 'CCCCNOT')
c4x.cry(pi/8, 0 ,4)
c4x.cx(0, 1)
c4x.cry(-pi/8, 1, 4)
c4x.cx(0, 1)
c4x.cry(pi/8, 1, 4)
c4x.cx(1, 2)
c4x.cry(-pi/8, 2, 4)
c4x.cx(0, 2)
c4x.cry(pi/8, 2, 4)
c4x.cx(1, 2)
c4x.cry(-pi/8, 2, 4)
c4x.cx(0, 2)
c4x.cry(pi/8, 2, 4)
c4x.cx(2, 3)
c4x.cry(-pi/8, 3, 4)
c4x.cx(0, 3)
c4x.cry(pi/8, 3, 4)
c4x.cx(1, 3)
c4x.cry(-pi/8, 3, 4)
c4x.cx(0, 3)
c4x.cry(pi/8, 3, 4)
c4x.cx(2, 3)
c4x.cry(-pi/8, 3, 4)
c4x.cx(0, 3)
c4x.cry(pi/8, 3, 4)
c4x.cx(1, 3)
c4x.cry(-pi/8, 3, 4)
c4x.cx(0, 3)
c4x.cry(pi/8, 3, 4)
ccccnot = c4x.to_gate()

permutation10 = QuantumCircuit(10, name = 'P10')
permutation10.swap(9, 7)
permutation10.swap(8, 5)
permutation10.swap(7, 5)
permutation10.swap(6, 3)
permutation10.swap(5, 3)
permutation10.swap(4, 0)
permutation10.swap(2, 1)
p10 = permutation10.to_gate()

permutation8 = QuantumCircuit(18, name = 'P8')
permutation8.cx(0, 11)
permutation8.cx(1, 10)
permutation8.cx(2, 13)
permutation8.cx(3, 15)
permutation8.cx(4, 17)
permutation8.cx(5, 12)
permutation8.cx(6, 14)
permutation8.cx(7, 16)
p8 = permutation8.to_gate()

leftshift1 = QuantumCircuit(10, name = 'LS1')
leftshift1.swap(9, 8)
leftshift1.swap(8, 7)
leftshift1.swap(7, 6)
leftshift1.swap(6, 5)
leftshift1.swap(4, 3)
leftshift1.swap(3, 2)
leftshift1.swap(2, 1)
leftshift1.swap(1, 0)
ls1 = leftshift1.to_gate()

leftshift2 = QuantumCircuit(10, name = 'LS2')
leftshift2.swap(9, 7)
leftshift2.swap(8, 6)
leftshift2.swap(7, 5)
leftshift2.swap(6, 5)
leftshift2.swap(4, 2)
leftshift2.swap(3, 1)
leftshift2.swap(2, 0)
leftshift2.swap(1, 0)
ls2 = leftshift2.to_gate()

leftshiftm2 = QuantumCircuit(10, name = 'LS-2')
leftshiftm2.swap(0, 1)
leftshiftm2.swap(0, 2)
leftshiftm2.swap(1, 3)
leftshiftm2.swap(2, 4)
leftshiftm2.swap(5, 6)
leftshiftm2.swap(5, 7)
leftshiftm2.swap(6, 8)
leftshiftm2.swap(7, 9)
lsm2 = leftshiftm2.to_gate()

leftshift3 = QuantumCircuit(10, name = 'LS3')
leftshift3.swap(9, 6)
leftshift3.swap(8, 5)
leftshift3.swap(7, 6)
leftshift3.swap(6, 5)
leftshift3.swap(4, 1)
leftshift3.swap(3, 0)
leftshift3.swap(2, 1)
leftshift3.swap(1, 0)
ls3 = leftshift3.to_gate()

initialpermutation = QuantumCircuit(8, name = 'IP')
initialpermutation.swap(7, 4)
initialpermutation.swap(7, 6)
initialpermutation.swap(6, 2)
initialpermutation.swap(3, 2)
initialpermutation.swap(2, 0)
initialpermutation.swap(1, 0)
ip = initialpermutation.to_gate()

initialpermutationm1 = QuantumCircuit(8, name = 'IP-1')
initialpermutationm1.swap(7, 6)
initialpermutationm1.swap(7, 4)
initialpermutationm1.swap(4, 3)
initialpermutationm1.swap(3, 1)
initialpermutationm1.swap(2, 1)
initialpermutationm1.swap(1, 0)
ipm1 = initialpermutationm1.to_gate()

expantionpermutation = QuantumCircuit(8, name = 'E/P')
expantionpermutation.cx(0, 7)
expantionpermutation.cx(3, 6)
expantionpermutation.cx(2, 5)
expantionpermutation.cx(1, 4)
expantionpermutation.cx(7, 0)
expantionpermutation.cx(6, 3)
expantionpermutation.cx(5, 2)
expantionpermutation.cx(4, 1)
expantionpermutation.cx(7, 1)
expantionpermutation.cx(6, 0)
expantionpermutation.cx(5, 3)
expantionpermutation.cx(4, 2)
ep = expantionpermutation.to_gate()

sbox0 = QuantumCircuit(6, name = 'S0/P4')
sbox0.append(ccccnot, [0, 1, 2, 3, 4])
sbox0.append(ccccnot, [0, 1, 2, 3, 5])
sbox0.append(cccnot, [0, 2, 3, 4])
sbox0.ccx(2, 3, 0)
sbox0.ccx(1, 3, 0)
sbox0.ccx(0, 3, 1)
sbox0.cx(3, 1)
sbox0.cx(2, 0)
sbox0.cx(1, 4)
sbox0.cx(0, 5)
sbox0.x(4)
s0 = sbox0.to_gate()

sbox1 = QuantumCircuit(6, name = 'S1/P4')
sbox1.append(ccccnot, [0, 1, 2, 3, 5])
sbox1.append(cccnot, [1, 2, 3, 5])
sbox1.append(cccnot, [0, 2, 3, 4])
sbox1.append(cccnot, [0, 1, 3, 5])
sbox1.ccx(1, 3, 2)
sbox1.ccx(0, 3, 4)
sbox1.cx(3, 2)
sbox1.cx(3, 1)
sbox1.ccx(0, 2, 4)
sbox1.cx(2, 5)
sbox1.ccx(0, 1, 4)
sbox1.ccx(0, 1, 5)
sbox1.cx(1, 4)
sbox1.cx(0, 5)
s1 = sbox1.to_gate()

switch = QuantumCircuit(8, name = 'SW')
switch.swap(0, 4)
switch.swap(1, 5)
switch.swap(2, 6)
switch.swap(3, 7)
sw = switch.to_gate()

#ここから検証用
#以下二つは改良前
sbox0v1 = QuantumCircuit(8, name = 'S0/P4(old)')
sbox0v1.ccx(0, 1, 6)
sbox0v1.ccx(2, 3, 7)
sbox0v1.ccx(7, 6, 4)
sbox0v1.ccx(2, 3, 7)
sbox0v1.ccx(0, 1, 6)
sbox0v1.ccx(0, 1, 6)
sbox0v1.ccx(2, 3, 7)
sbox0v1.ccx(7, 6, 5)
sbox0v1.ccx(2, 3, 7)
sbox0v1.ccx(0, 1, 6)
sbox0v1.ccx(0, 2, 6)
sbox0v1.ccx(6, 3, 4)
sbox0v1.ccx(0, 2, 6)
sbox0v1.ccx(2, 3, 5)
sbox0v1.ccx(2, 3, 4)
sbox0v1.ccx(1, 3, 5)
sbox0v1.ccx(1, 3, 4)
sbox0v1.ccx(3, 0, 4)
sbox0v1.cx(3, 4)
sbox0v1.cx(2, 5)
sbox0v1.cx(1, 4)
sbox0v1.cx(0, 5)
sbox0v1.x(4)
s0v1 = sbox0v1.to_gate()

sbox1v1 = QuantumCircuit(8, name = 'S0/P4(old)')
sbox1v1.ccx(0, 1, 6)
sbox1v1.ccx(2, 3, 7)
sbox1v1.ccx(6, 7, 5)
sbox1v1.ccx(2, 3, 7)
sbox1v1.ccx(0, 1, 6)
sbox1v1.ccx(2, 1, 6)
sbox1v1.ccx(6, 3, 5)
sbox1v1.ccx(1, 2, 6)
sbox1v1.ccx(2, 0, 6)
sbox1v1.ccx(6, 3, 4)
sbox1v1.ccx(0, 2, 6)
sbox1v1.ccx(1, 0, 6)
sbox1v1.ccx(6, 3, 4)
sbox1v1.ccx(0, 1, 6)
sbox1v1.ccx(1, 0, 6)
sbox1v1.ccx(6, 3, 5)
sbox1v1.ccx(0, 1, 6)
sbox1v1.ccx(3, 1, 5)
sbox1v1.ccx(3, 0, 5)
sbox1v1.ccx(0, 3, 4)
sbox1v1.cx(3, 5)
sbox1v1.cx(3, 4)
sbox1v1.ccx(2, 0, 4)
sbox1v1.cx(2, 5)
sbox1v1.ccx(0, 1, 5)
sbox1v1.ccx(0, 1, 4)
sbox1v1.cx(1, 4)
sbox1v1.cx(0, 5)
s1v1 = sbox1v1.to_gate()



mode = 1 #0なら暗号化 1なら復号化



SDES = QuantumCircuit(28, 8)
SDES.h(0)
SDES.h(1)
SDES.h(2)
SDES.h(3)
SDES.h(4)
SDES.h(5)
SDES.h(6)
SDES.h(7)
SDES.h(0 + 16)
SDES.h(1 + 16)
SDES.h(2 + 16)
SDES.h(3 + 16)
SDES.h(4 + 16)
SDES.h(5 + 16)
SDES.h(6 + 16)
SDES.h(7 + 16)
SDES.h(8 + 16)
SDES.h(9 + 16)

SDES.barrier()
SDES.append(ip, [0, 1, 2, 3, 4, 5, 6, 7])
SDES.append(p10, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
if mode == 0:
    SDES.append(ls1, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
elif mode == 1:
    SDES.append(ls3, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

SDES.cx(0, 8)
SDES.cx(1, 9)
SDES.cx(2, 10)
SDES.cx(3, 11)

SDES.append(ep, [8, 9, 10, 11, 12, 13, 14, 15])
SDES.append(p8, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 8, 9, 10, 11, 12, 13, 14, 15])

#検証したい組み合わせにS0、S1を変更してください
SDES.append(s0, [12, 13, 14, 15, 7, 4])
#SDES.append(s0v1, [12, 13, 14, 15, 7, 4, 26, 27])
SDES.append(s1, [8, 9, 10, 11, 6, 5])
#SDES.append(s1v1, [8, 9, 10, 11, 6, 5, 26, 27])


SDES.reset(8)
SDES.reset(9)
SDES.reset(10)
SDES.reset(11)
SDES.reset(12)
SDES.reset(13)
SDES.reset(14)
SDES.reset(15)
if mode == 0:
    SDES.append(ls2, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
elif mode == 1:
    SDES.append(lsm2, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

SDES.append(sw, [0, 1, 2, 3, 4, 5, 6, 7])
SDES.cx(0, 8)
SDES.cx(1, 9)
SDES.cx(2, 10)
SDES.cx(3, 11)

SDES.append(ep, [8, 9, 10, 11, 12, 13, 14, 15])
SDES.append(p8, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 8, 9, 10, 11, 12, 13, 14, 15])

#検証したい組み合わせにS0、S1を変更してください
SDES.append(s0, [12, 13, 14, 15, 7, 4])
#SDES.append(s0v1, [12, 13, 14, 15, 7, 4, 26, 27])
SDES.append(s1, [8, 9, 10, 11, 6, 5])
#SDES.append(s1v1, [8, 9, 10, 11, 6, 5, 26, 27])

SDES.append(ipm1, [0, 1, 2, 3, 4, 5, 6, 7])

SDES.barrier()

SDES.measure(0, 0)
SDES.measure(1, 1)
SDES.measure(2, 2)
SDES.measure(3, 3)
SDES.measure(4, 4)
SDES.measure(5, 5)
SDES.measure(6, 6)
SDES.measure(7, 7)

#backend = Aer.get_backend('qasm_simulator')
#shots = 1
#results = execute(SDES, backend=backend, shots=shots, memory=True).result()
#bits = "".join(results.get_memory())
#print(bits)

#SDES.draw(output = 'mpl')



#sdesd1 = SDES.decompose()
#sdesd2 = sdesd1.decompose()
#print('depth = ', sdesd2.depth())
#sdesd1.draw(output = 'mpl')



timelist = list()
for x in range(100):
    backend = Aer.get_backend('qasm_simulator')
    shots = 1

    results = execute(SDES, backend=backend, shots=shots, memory=True).result()
    timelist.append(results.time_taken)
    
print(timelist)