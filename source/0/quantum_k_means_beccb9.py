# https://github.com/benkoehlL/Qiskit_Playground/blob/2a9ede4d0ce3cc9718dccb6ff06538d0ec32824a/quantum_k_means.py
'''
This program realises the qunatum k-means algorithm
'''

import numpy as np
from qiskit import *
from qiskit.tools.visualization import plot_histogram
from matplotlib.pyplot import plot, draw, show

circuit_name = 'k_means'

backend = Aer.get_backend('qasm_simulator')
theta_list = [0.01, 0.02, 0.03, 0.04, 0.05,
             1.31, 1.32, 1.33, 1.34, 1.35]

qr = QuantumRegister(5,'q')
cr = ClassicalRegister(5, 'c')
qc = QuantumCircuit(qr, cr, name=circuit_name)

# define a loop to compute the distance between each pair of points
for i in range(len(theta_list)-1):
    for j in range(1,len(theta_list)-i):
        # set the parameters theta about different points
        theta_1 = theta_list[i]
        theta_2 = theta_list[i+j]

        qc.h(qr[2])
        qc.h(qr[1])
        qc.h(qr[4])
        qc.u3(theta_1, np.pi, np.pi, qr[1])
        qc.u3(theta_2, np.pi, np.pi, qr[4])
        qc.cswap(qr[2], qr[1], qr[4])
        qc.h(qr[2])

        qc.measure(qr[2], cr[2])
        qc.reset(qr)

        job = execute(qc, backend=backend, shots=1024)
        result = job.result()
        print(result.get_counts())
        print('theta_1: ', theta_1, '\t', 'theta_2: ', theta_2)
plot_histogram(result.get_counts())
