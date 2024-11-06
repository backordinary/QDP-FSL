# https://github.com/matteobarbieri00/Quantum/blob/099591b02cbf7a005d40d8856b9c5f6cb0237e54/PhaseFlip_QEC_3qubits.py
import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit import * 
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator

# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()

qr_sys = QuantumRegister(3)
qr_a = QuantumRegister(2)
cr0 = ClassicalRegister(1)
cr1 = ClassicalRegister(1)
circ = QuantumCircuit(qr_sys,qr_a,cr0,cr1)
#circ.x(qr_sys[0]) #1
#circ.h(qr_sys[0]) #+
circ.x(qr_sys[0]) #-
circ.h(qr_sys[0]) 
circ.cx(qr_sys[0],qr_sys[1])
circ.cx(qr_sys[0],qr_sys[2])
circ.h(qr_sys[0])
circ.h(qr_sys[1])
circ.h(qr_sys[2])
circ.h(qr_a[0])
circ.h(qr_a[1])
circ.cx(qr_sys[0],qr_a[0])
circ.cx(qr_sys[1],qr_a[0])
circ.cx(qr_sys[1],qr_a[1])
circ.cx(qr_sys[2],qr_a[1])
circ.h(qr_sys[0])
circ.h(qr_sys[1])
circ.h(qr_sys[2])
circ.h(qr_a[0])
circ.h(qr_a[1])
circ.measure(qr_a[0],cr0)
circ.measure(qr_a[1],cr1)
circ.draw('mpl')

#Running on device
from qiskit.tools.visualization import plot_histogram
IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q')
gcomp = provider.get_backend('ibmq_manila')
job = execute(circ,backend=gcomp, shots = 20000)
from qiskit.tools.monitor import job_monitor
job_monitor(job)
result = job.result()
plot_histogram(result.get_counts(circ))
counts = result.get_counts(circ)
print(counts)
with open('3qbfi-.txt', 'w') as file:
    #file.write(counts)
    for gate_i in gcomp.properties().gates:
        #print("{} gate on qubits {} error rate is {}{}".format(gate_i.name, gate_i.qubits, gate_i.parameters[0].value, gate_i.parameters[0].unit))
        file.write("{} gate on qubits {} error rate is {}{}".format(gate_i.name, gate_i.qubits, gate_i.parameters[0].value, gate_i.parameters[0].unit))
        file.write('\n')
