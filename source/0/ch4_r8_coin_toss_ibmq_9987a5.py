# https://github.com/PacktPublishing/Quantum-Computing-in-Practice-with-Qiskit-and-IBM-Quantum-Experience/blob/938123d051c5bab72110011b3a05e515bb69ca09/Chapter04/ch4_r8_coin_toss_IBMQ.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2020

@author: hassi
"""

from qiskit import QuantumCircuit, execute
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from IPython.core.display import display

print("Getting provider...")
if not IBMQ.active_account():
    IBMQ.load_account()
provider = IBMQ.get_provider()

print("Ch 4: Quantum coin toss on IBM Q backend")
print("----------------------------------------")

qc = QuantumCircuit(2, 2)

qc.h(0)
qc.cx(0,1)
qc.measure([0,1],[0,1])

display(qc.draw('mpl'))

from qiskit.providers.ibmq import least_busy
backend = least_busy(provider.backends(n_qubits=5, operational=True, simulator=False))
print(backend.name())

job = execute(qc, backend, shots=1000)
job_monitor(job)

result = job.result()
print(result)
counts = result.get_counts(qc)
from qiskit.tools.visualization import plot_histogram
display(plot_histogram(counts))
