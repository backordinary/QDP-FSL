# https://github.com/abhaduri77/k-clique-code/blob/e37c7e60a0153e71deae5903e886bb738f26ec95/triangle_4vertex.py

from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit,execute,BasicAer,IBMQ
#from qiskit.tools.visualization import matplotlib_circuit_drawer as circuit_drawer
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
import matplotlib.pyplot as plt
#from qiskit import compile
import math
from qiskit import *
pi=math.pi
import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit import IBMQ

token = 'ae54a2937f60a5daeca95a84237ab43fbdc7b4c3e7a3fbe3a249f9c2e50e18a278f3c65340d50560403275052c5e51fa0f5d4558f94b769fef629801039196a3'
url = 'https://quantumexperience.ng.bluemix.net/qx/account/advanced'


IBMQ.save_account(token, overwrite=True)




n=6
ctrl=QuantumRegister(n,'ctrl')

anc=QuantumRegister(3,'anc')
tgt=QuantumRegister(2,'tgt')
c1=ClassicalRegister(n,'c1')

circuit=QuantumCircuit(ctrl,anc,tgt,c1)


circuit.h(ctrl)
circuit.x(tgt[1])
circuit.h(tgt[1])
#circuit.h(tgt)


#1st combination 000110

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

#2nd combination 001011
circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[3])


circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')
circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')


circuit.mct(anc,tgt[0],None,mode='noancilla')


circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')
circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')





circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[3])
#3rd combbination 011011

circuit.x(ctrl[0])
circuit.x(ctrl[3])



circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')
circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')


circuit.mct(anc,tgt[0],None,mode='noancilla')


circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')
circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')
circuit.x(ctrl[0])
circuit.x(ctrl[3])

#4th combination 000111
circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[2])


circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')



circuit.mct(anc,tgt[0],None,mode='noancilla')



circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')


circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[2])



circuit.cx(tgt[0],tgt[1])


#Reverse latch
#4th combination 000111
circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[2])

circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')



circuit.mct(anc,tgt[0],None,mode='noancilla')



circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')


circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[2])

#3rd combbination 011011

circuit.x(ctrl[0])
circuit.x(ctrl[3])

circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')
circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')


circuit.mct(anc,tgt[0],None,mode='noancilla')


circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')
circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')

circuit.x(ctrl[0])
circuit.x(ctrl[3])
#2nd combination 001011
circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[3])


circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')
circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')


circuit.mct(anc,tgt[0],None,mode='noancilla')


circuit.mct(ctrl[2:5],anc[1],None,mode='noancilla')
circuit.mct(ctrl[0:3],anc[0],None,mode='noancilla')



circuit.x(ctrl[0])
circuit.x(ctrl[1])
circuit.x(ctrl[3])

#1st combination 000110


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



#phase flip gate
#circuit.cz(ctrl,ctrl[5])
#circuit.z(ctrl)

#amplification


circuit.h(ctrl)
circuit.x(ctrl)
circuit.h(ctrl[5])


circuit.mct(ctrl[0:4],ctrl[5],None,mode='noancilla')


circuit.h(ctrl[5])
circuit.x(ctrl)
circuit.h(ctrl)

circuit.h(tgt[1])
circuit.x(tgt[1])
# Insert a barrier before measurement
circuit.barrier()
# Measure all of the qubits in the standard basis
for i in range(n):
    circuit.measure(ctrl[i], c1[i])
########################

###############################################################
# Set up the API and execute the program.
###############################################################


# First version: simulator
provider = IBMQ.load_account()
#backend = provider.get_backend('ibmq_16_melbourne')
my_provider = IBMQ.get_provider()

device = my_provider.get_backend(name='ibmq_qasm_simulator')
job=execute(circuit,backend=device,shots=8192)

result=job.result()
print(job.status())
count=result.get_counts()
data1=result.data()
print(result.get_counts(circuit))


print((circuit.decompose()).count_ops())
print(circuit.size())
print(circuit.depth())
print(circuit.clifford_2_1())