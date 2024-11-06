# https://github.com/flyjgh/quantum/blob/95d078e3855bb9126079b32231d963bf883aa204/src/qt.py
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ

IBMQ.save_account('9ffab3f25085a5663f948657d7acc06c2b049ce98728bbb117794e57037d71fad138023e02f2e55ce90acb37545bb4d6f4813485e28f319a090272181c649402')

IBMQ.load_account()

nQbits = 2
nBits = 2
qbits = QuantumRegister(nQbits)
bits = ClassicalRegister(nBits)

circuit = qiskit.QuantumCircuit(qbits, bits)

circuit.h(qbits[0])
circuit.x(qbits[1])
circuit.cx(qbits[0],qbits[1])
circuit.measure(qbits, bits)

circuit.draw()
# for backend in Aer.backends():
#     print(backend)
provider = IBMQ.get_provider('ibm-q')
# qcomputer = provider.get_backend('ibmq_16_melbourne')
qcomputer = provider.get_backend('ibmq_athens')
simulator = Aer.get_backend('qasm_simulator')
execution = execute(circuit, backend=simulator, shots=1000)
job_monitor(execution)
result = execution.result()
plot_histogram(result.get_counts(circuit))
