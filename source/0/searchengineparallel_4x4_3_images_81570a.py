# https://github.com/CleverCracker/Quantum_Image_Based_Search_Engine/blob/42baeb62714d685a011213fe209cab208154e301/SearchEngineParallel_4x4_3_Images.py
from qiskit import Aer, IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.monitor import job_monitor
import numpy as np

imageDir = "images/4x4/"
imageNames = ["00","01","02","03","10","11","12","13","20","21","22","23","30","31","32","33"]
imageExt = ".jpg"
result = []
data = np.loadtxt('data_4x4.csv', delimiter=',', dtype=np.complex128)

"""
Result Measure Togeather
"""

targetQubit_1 = QuantumRegister(1, 'target_1')
targetQubit_2 = QuantumRegister(1, 'target_2')

ref_1 = QuantumRegister(5, 'ref_1')
ref_2 = QuantumRegister(5, 'ref_2')

original = QuantumRegister(5, 'original')

anc = QuantumRegister(3, 'anc')

c = ClassicalRegister(2)

qc = QuantumCircuit(targetQubit_1, targetQubit_2,
                    ref_1, ref_2, original, anc, c)

numOfShots = 100

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_qasm_simulator')

index = 1


qc.initialize(data[3], [2, 3, 4, 5, 6, 17])
qc.initialize(data[2], [7, 8, 9, 10, 11, 18])
qc.initialize(data[1], [12, 13, 14, 15, 16, 19])

qc.tdg(targetQubit_1[0])
qc.h(targetQubit_1[0])

qc.tdg(targetQubit_2[0])
qc.h(targetQubit_2[0])

for i in range(len(ref_1)):
    qc.cswap(targetQubit_1[0], ref_1[i], original[i])

for i in range(len(ref_2)):
    qc.cswap(targetQubit_2[0], ref_2[i], original[i])

qc.h(targetQubit_1[0])
qc.tdg(targetQubit_1[0])

qc.h(targetQubit_2[0])
qc.tdg(targetQubit_2[0])
targetQubit_2[0]
targetQubit_1[0]
qc.measure([targetQubit_1[0], targetQubit_2[0]], c)


job = execute(qc, backend, shots=numOfShots)

job_monitor(job)
counts = job.result().get_counts()

print(counts)


"""
Result Target Meausre  
"""
"""

"""
targetQubit_1 = QuantumRegister(1, 'target_1')
targetQubit_2 = QuantumRegister(1, 'target_2')

ref_1 = QuantumRegister(5, 'ref_1')
ref_2 = QuantumRegister(5, 'ref_2')

original = QuantumRegister(5, 'original')

anc = QuantumRegister(1, 'anc')

c = ClassicalRegister(1)

qc = QuantumCircuit(targetQubit_1, targetQubit_2,
                    ref_1, ref_2, original, anc)

numOfShots = 100

backend = Aer.get_backend('statevector_simulator')

qc.initialize(data[2], [2, 3, 4, 5, 6, 17])
qc.initialize(data[1], [7, 8, 9, 10, 11, 17])
qc.initialize(data[1], [12, 13, 14, 15, 16, 17])

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

job = execute(qc, backend, shots=numOfShots)


statevector = job.result().get_statevector()

backends = Aer.get_backend('qasm_simulator')

_qc = QuantumCircuit(targetQubit_1, targetQubit_2,
                    ref_1, ref_2, original, anc, c)
_qc.initialize(statevector,range(18))
_qc.measure(0,c)

result = execute(_qc,backends,shots=numOfShots)
count1 = result.result().get_counts()

_qc = QuantumCircuit(targetQubit_1, targetQubit_2,
                    ref_1, ref_2, original, anc, c)
_qc.initialize(statevector,range(18))

_qc.measure(1,c)

result = execute(_qc,backends,shots=numOfShots)
count2 = result.result().get_counts()

print(count1,count2)