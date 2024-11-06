# https://github.com/NorbertZare/Quantum-Computer-VS.-Personal-Computer/blob/3c4540f6e322c885d8b041ef9597ed77478a45b5/Quantum.py
from qiskit import *
import random
from qiskit.tools.monitor import job_monitor

IBMQ.save_account('your token here')

IBMQ.load_account()

def initial_circuit():
    circuit = QuantumRegister(1, 'circuit')
    measure = ClassicalRegister(1, 'result')
    qc = QuantumCircuit(circuit, measure)
    return qc, circuit, measure


fon=random.randint(0,1) # 1 means flip the coin & 0 means do nothing

qc, circuit, measure = initial_circuit()

qc.h(circuit[0])

if fon == 0:
    qc.i(circuit[0])
if fon == 1:
    qc.x(circuit[0])

qc.h(circuit[0])

qc.measure(circuit, measure)


provider = IBMQ.get_provider('ibm-q')
backend_real = provider.get_backend('ibmq_belem')
job_real = execute(qc, backend_real, shots=100)
job_monitor(job_real)
res_real = job_real.result().get_counts()
print(res_real)