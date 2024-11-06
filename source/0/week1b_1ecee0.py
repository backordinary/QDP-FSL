# https://github.com/amarjahin/IBMQuantumChallenge2020/blob/152776bbd837fc442f534ec5bc02ba39b11ac50d/week1b.py
# Initialization
import matplotlib.pyplot as plt
import numpy as np

# Importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

# Import basic plot tools
from qiskit.tools.visualization import plot_histogram

backend = Aer.get_backend('qasm_simulator')
prob_of_ans = []

data = [0,1,2,3,4,5,6]
oracle = [7]
cr =   [0,1,2,3,4,5,6]

for x in range(15):
    qc = QuantumCircuit(len(data) + len(oracle), len(cr))
    qc.x(oracle[0])
    qc.h(data)
    for i in range(x):
        # Apply oracle 
        qc.h(oracle[0])
        qc.mcx(data, oracle)
        qc.h(oracle[0])
        qc.barrier()

        # Apply dispersion 
        qc.h(data)
        qc.x(data)
        qc.barrier()
        qc.h(data[6])
        qc.mcx(data[0:6], data[6])
        qc.h(data[6])
        qc.barrier()
        qc.x(data)
        qc.h(data)
        qc.barrier()

    qc.measure(data,cr)

    # print(qc.draw())

    job = execute(qc, backend=backend, shots=1000, seed_simulator=12345, backend_options={"fusion_enable":True})
    result = job.result()
    count = result.get_counts()
    answer = count['1111111']
    prob_of_ans.append(answer)


iteration = [i for i in range(15)]
correct = prob_of_ans
plt.bar(iteration, correct)
plt.xlabel('# of iteration')
plt.ylabel('# of times the solution was obtained')


# print(qc.draw())


