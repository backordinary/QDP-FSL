# https://github.com/abhaduri77/k-clique-code/blob/e37c7e60a0153e71deae5903e886bb738f26ec95/triangle_ideal.py
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:36:37 2021

@author: hp
"""

from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit,execute,BasicAer,IBMQ
#from qiskit.tools.visualization import matplotlib_circuit_drawer as circuit_drawer
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
import matplotlib.pyplot as plt
#from qiskit import compile
import math

pi=math.pi
from qiskit import IBMQ

token = 'ae54a2937f60a5daeca95a84237ab43fbdc7b4c3e7a3fbe3a249f9c2e50e18a278f3c65340d50560403275052c5e51fa0f5d4558f94b769fef629801039196a3'
url = 'https://quantumexperience.ng.bluemix.net/qx/account/advanced'


IBMQ.save_account(token, overwrite=True)



n=6
ctrl=QuantumRegister(n,'ctrl')

anc=QuantumRegister(3,'anc')
tgt=QuantumRegister(1,'tgt')
c1=ClassicalRegister(n,'c1')

circuit=QuantumCircuit(ctrl,anc,tgt,c1)


circuit.h(ctrl)
circuit.x(tgt[0])
circuit.h(tgt[0])
#circuit.h(tgt)

#2nd combination 000110
circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[2])
circuit.x(ctrl[5])


circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')
circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')
circuit.mct([ctrl[0],ctrl[1],ctrl[4],ctrl[5]],anc[2],None,mode='noancilla')

circuit.mct(anc,tgt[0],None,mode='noancilla')

circuit.mct([ctrl[0],ctrl[1],ctrl[4],ctrl[5]],anc[2],None,mode='noancilla')
circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')
circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')



circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[2])
circuit.x(ctrl[5])



#amplification


circuit.h(ctrl)
circuit.x(ctrl)
circuit.h(ctrl[5])


circuit.mct(ctrl[0:4],ctrl[5],None,mode='noancilla')

#circuit.ccx(ctrl[3],ctrl[4],ctrl[5])
circuit.h(ctrl[5])
circuit.x(ctrl)
circuit.h(ctrl)

circuit.h(tgt[0])
circuit.x(tgt[0])
# Insert a barrier before measurement
circuit.barrier()
# Measure all of the qubits in the standard basis
for i in range(n):
    circuit.measure(ctrl[i], c1[i])
########################

###############################################################
# Set up the API and execute the program.
###############################################################



provider = IBMQ.load_account()

my_provider = IBMQ.get_provider()


device = my_provider.get_backend(name='ibmq_qasm_simulator')
job=execute(circuit,backend=device,shots=8192)
result = job.result()
final = result.get_counts()
result_freqs = {'000000':0, '000001':0, '000010':0, '000011':0, '000100':0,'000101':0,'000110':0,'000111':0,'001000':0,'001001':0,'001010':0,'001011':0,'001100':0,'001101':0,'001110':0,'001111':0,'010000':0,'010001':0,'010010':0,'010011':0,'010100':0,'010101':0,
                '010110':0,'010111':0,'011000':0,'011001':0,'011010':0,'011011':0,'011100':0,'011101':0,'011110':0,'011111':0,'100000':0,'100001':0,'100010':0,'100011':0,'100100':0,'100011':0,'100100':0,'100101':0,'100110':0,'100111':0,'101000':0,'101001':0,'101010':0,
                '101011':0,'101100':0,'101101':0,'101110':0,'101111':0,'110000':0,'110001':0,'110010':0,'110011':0,'110100':0,'110101':0,'110110':0,'110111':0,'111000':0,'111001':0,'111010':0,'111011':0,'111100':0,'111101':0,'111110':0,'111111':0
                }
for key in final.keys():
    freq_key = key[0:6]
    result_freqs[freq_key] = result_freqs[freq_key] +final[key] 
    D = result_freqs
    plt.figure(figsize=(20,10))
plt.bar(range(len(D)),list(D.values()), align='center')
plt.xlabel('States')
plt.ylabel('Frequency')
plt.xticks(range(len(D)), list(D.keys()))
plt.xticks(rotation=90)
print((circuit.decompose()).count_ops())
print((circuit.decompose()).size())
print((circuit.decompose()).depth())
