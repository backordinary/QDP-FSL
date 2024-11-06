# https://github.com/CleverCracker/Quantum_Image_Based_Search_Engine/blob/42baeb62714d685a011213fe209cab208154e301/SearchEngineParallel_4x4_4_Images.py
from qiskit import Aer, IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
import numpy as np
from qiskit.tools.monitor.job_monitor import job_monitor

IBMQ.load_account()

imageDir = "images/4x4/"
imageNames = ["00","01","02","03","10","11","12","13","20","21","22","23","30","31","32","33"]
imageExt = ".jpg"

result = []
data = np.loadtxt('data_4x4.csv', delimiter=',', dtype=np.complex128)

# ! EXCUTION HERE

targetQubit_1 = QuantumRegister(1, 'target_1')
targetQubit_2 = QuantumRegister(1, 'target_2')
targetQubit_3 = QuantumRegister(1, 'target_3')

ref_1 = QuantumRegister(5, 'ref_1')
ref_2 = QuantumRegister(5, 'ref_2')
ref_3 = QuantumRegister(5, 'ref_3')

original = QuantumRegister(5, 'original')

anc = QuantumRegister(4, 'anc')

c = ClassicalRegister(3)

qc = QuantumCircuit(targetQubit_1, targetQubit_2,targetQubit_3,ref_1, ref_2,ref_3, original, anc,c)

numOfShots = 100
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_qasm_simulator')
# backend = provider.get_backend('simulator_statevector')
# backend = Aer.get_backend('statevector_simulator')

qc.initialize(data[0], [3, 4, 5, 6, 7,23 ])
qc.initialize(data[1], [8, 9, 10, 11, 12,24 ])
qc.initialize(data[3], [13, 14, 15, 16, 17,25 ])
qc.initialize(data[2], [18, 19, 20, 21, 22,26 ])

qc.tdg(targetQubit_1[0])
qc.h(targetQubit_1[0])
for i in range(len(ref_1)):
    qc.cswap(targetQubit_1[0], ref_1[i], original[i])
qc.h(targetQubit_1[0])
qc.tdg(targetQubit_1[0])

qc.tdg(targetQubit_2[0])
qc.h(targetQubit_2[0])
for i in range(len(ref_2)):
    qc.cswap(targetQubit_2[0], ref_2[i], original[i])
qc.h(targetQubit_2[0])
qc.tdg(targetQubit_2[0])

qc.tdg(targetQubit_3[0])
qc.h(targetQubit_3[0])
for i in range(len(ref_3)):
    qc.cswap(targetQubit_3[0], ref_3[i], original[i])
qc.h(targetQubit_3[0])
qc.tdg(targetQubit_3[0])

qc.measure([0,1,2], c)

job = execute(qc, backend, shots=numOfShots)
job_monitor(job)

result = job.result()
result.get_counts()





"""
    Measurement Diffrent Target with Diffrent Measurement
"""

# statevector = job.result().get_statevector()

# ! Target 1
#  
# backends = Aer.get_backend('qasm_simulator')
# _qc = QuantumCircuit(targetQubit_1, targetQubit_2,targetQubit_3,
#                     ref_1, ref_2, ref_3, original, anc, c)
# _qc.initialize(statevector,range(27))
# _qc.measure(0,c)

# result = execute(_qc,backends,shots=numOfShots)
# count1 = result.result().get_counts()

# ! Target 2

# _qc = QuantumCircuit(targetQubit_1, targetQubit_2,targetQubit_3,
#                     ref_1, ref_2, ref_3, original, anc, c)
# _qc.initialize(statevector,range(27))
# _qc.measure(1,c)

# result = execute(_qc,backends,shots=numOfShots)
# count2 = result.result().get_counts()

# ! Target 3

# _qc = QuantumCircuit(targetQubit_1, targetQubit_2,targetQubit_3,
#                     ref_1, ref_2, ref_3, original, anc, c)
# _qc.initialize(statevector,range(27))
# _qc.measure(2,c)

# result = execute(_qc,backends,shots=numOfShots)
# count3 = result.result().get_counts()

# Simularties Show

# print(count1,count2,count3)