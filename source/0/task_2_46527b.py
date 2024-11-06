# https://github.com/Geomon-Joshy/qosf_mentornship_task_2/blob/675352f340fac143cc7650a25fff874768fc2a66/task_2.py
from qiskit import QuantumCircuit,Aer,assemble #importing required libraries
from qiskit.visualization import plot_histogram
from numpy import array
import random
sim = Aer.get_backend('aer_simulator')  #importing the simulator
list1=[0,1,2,3] #creating a list for selecting a random number
a = random.choice(list1) #selcting a random number
n=4 # setting the number of bits.
r=array([0]*n) #defining an array to store the random 4-bit number
if a==1: #creating a 4-bit number based on the ranom number 'a'
        r[0]=1
for i in range(a):
    if a>=1:
        r[i]=a%2
        a=a/2
    i=i+1
if r[0] == r[1]:
    r[3]=r[2]=1-r[1]
else:
    r[3] = r[1]
    r[2] = r[0]
qc=QuantumCircuit(4,4)
for i in range(n): #converting the 4-bit array to a QuantumCircuit
    if r[i]==1:
        qc.x(i)
qc.barrier()
for i in range(n-1):  #doing some quantum operatios.
    qc.cx(i,i+1)
qc.barrier()
for i in range(n): #measuring the results
    qc.measure(i,i)
qobj = assemble(qc) #simulating the circuit 
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts) #ploting a histogram of the value.
