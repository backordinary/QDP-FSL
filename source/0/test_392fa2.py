# https://github.com/ironmanaudi/Machine-Learning-for-QEM/blob/85c35e59ae8480d67f8e631c89f5ce8cb7091a7e/test.py
from qiskit.circuit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import execute
from qiskit import IBMQ
from qiskit import BasicAer

#IBMQ.save_account('83a95fc7efba05f250ed50ff6bdf1638541ebd4f9e46396f26c8dc12f15b2331ea24d9db4f59c30b6e397c2268e40ddc1dae4772091baa73d7e99c91e8d8c56b')
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_quito')

qr = QuantumRegister(3)
cr =ClassicalRegister(3)
qc = QuantumCircuit(qr,cr)

qc.h(0)
qc.cx(0,1)
qc.cx(0,2)
qc.measure(qr[0:],cr[0:])

for _ in range(20):
    job = execute(qc,backend = backend,shots = 1024)
    counts = job.result().get_counts()
    print(counts)




