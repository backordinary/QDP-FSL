# https://github.com/braqiiit/Nonlinearity-Estimation/blob/08a7a883971e8330aed693b54617ed6966b58140/nl_estimate_quantum.py
'''This is the python code that estimates the interval in which 
the nonlinearity of the function given as truth table lies using 
a purely quantum method.
        1. Please provide the file containing the function values 
    as data.txt or change accordingly in Line__.'''


import sys
from qiskit import *
from qiskit import IBMQ
from qiskit import BasicAer
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.circuits.gates import mcmt,mct
from math import *
import qconfig


#n = int(sys.argv[1])
threshold = float(sys.argv[1])
delta = float(sys.argv[2])
vflag = int(sys.argv[3])

#print('delta : ', delta);
#print('thresh : ', threshold);

vprint = print if vflag else lambda *a, **k:None

IBMQ.save_account(qconfig.APItoken, overwrite=True)
IBMQ.load_accounts()

backend = IBMQ.get_backend('ibmq_qasm_simulator')


def swap(qc,q1,q2,anc):
    ln = range(len(q1))
    for i in ln:
        qc.cx(q1[i],q2[i])
        qc.cx(q2[i],q1[i])
        qc.cx(q1[i],q2[i])

s=''
dat_file = open('data.txt','r')
n = int(dat_file.readline())

#delta = 100
#threshold = 100
#while delta > 1 or delta <=0:
#    delta = float(input('Please enter the maximum error allowed (value between 0 and 1 excluding 0): \n'))
#while threshold >=1 or threshold < (1/2**n):
#    threshold = float(input('Please enter the threshold below which the fhat_max lies (value between 1/2^n and 1 excluding 1): '))
#print('delta : ', delta, '\n')
samples = ceil((400/(1-threshold))*log(1/delta))
#print('shots : ', samples)

for i in range(2**n):
    temp = dat_file.readline()
    s+= list(temp.split("\t"))[1][:1]
dat_file.close()

vprint('Creating and Extracting oracle')
oracle = TruthTableOracle(s)
vr = oracle.variable_register
vr2 = QuantumRegister(len(vr))
anc = QuantumRegister(len(vr))
otr = oracle.output_register
qc1 = QuantumCircuit(vr,vr2,anc,otr)
qc1.h(vr)
qc1.x(otr)
qc1.h(otr)
qc1.barrier()

qc=oracle.circuit

qc2 = QuantumCircuit(vr,vr2,anc,otr)
qc2.h(vr)
swap(qc2,vr,vr2,anc)
qc2.h(vr)

qc3 = QuantumCircuit(vr,vr2,anc,otr)
swap(qc3,vr,vr2,anc)
qc3.h(vr2)
qc3.h(otr)
qc3.x(otr)

for i in range(len(vr)):
    qc3.cx(vr[i],vr2[i])

qc3.x(otr)
qc3.x(vr2)
qc3.mct(vr2,otr[0],anc)
qc3.x(vr2)

for i in range(len(vr)):
    qc3.cx(vr[i],vr2[i])

qc3.barrier()

vprint('Building circuit')
circ = qc1+qc+qc2+qc+qc3

c = ClassicalRegister(len(otr))
circ.add_register(c)
circ.measure(otr,c)


sims = ceil(samples/8192)
vprint('Simulations : ', sims)
remshots = samples - ((sims-1)*8192)
shots = 8192


counts = [0]*sims
for i in range(sims):
    vprint('Running Sim ', i+1)
    vprint('Running in backend')
    if i+1==sims:
        shots = remshots
    job = execute(circ,backend,shots=shots)
    vprint('Waiting for end of execution and for obtaining results')
    result = job.result()
    vprint('Aquired Results')
    counts[i] = job.result().get_counts()

vprint('counting ends')

favcount = 0
for i in range(sims):
    favcount += counts[i]['0']
est = favcount/samples
#upper = (2**(n-1))*(1-sqrt(1-est))
#lower = (2**(n-1))*(1-sqrt(threshold))
upper = (1-sqrt(est))/2
lower = (1-threshold)/2
print('')

if lower<upper:
    print('nl(f) belongs to [', round(lower,6), ',', round(upper,6), ']')
    vprint('\nTotal number of quantum queries made : ', samples)
else:
    print('You have entered a wrong threshold. Please correct it.')
