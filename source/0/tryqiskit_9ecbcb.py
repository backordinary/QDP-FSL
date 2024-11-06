# https://github.com/klumel8/VQE/blob/b38c5b8625017c5c967ad262c7389fd184926a0c/Sasha/TryQiskit.py
import numpy as np

from qiskit import(
	QuantumCircuit,
	execute,
	Aer,
	IBMQ)
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

#IBMQ.save_account('c650f2a6e47504e201bd6e1422e89efbb3b1ac24264be980a0555746424cd5d8469e1511b86a10f0f3fd1581245d864bcbe03b0802ce48dff35adc57a44da5f6')
IBMQ.load_account()

provider = IBMQ.get_provider(group='open', project='main')
backend = provider.get_backend('ibmq_vigo')
#backend = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0,1)
circuit.measure([0,1],[0,1])

job = execute(circuit,backend, shots=1000)
result = job.result()

counts = result.get_counts()

print("\nTotal count for 00 and 11 are:",counts)
print(circuit.draw(output="text",filename=None))
plot_histogram(counts)
plt.show()