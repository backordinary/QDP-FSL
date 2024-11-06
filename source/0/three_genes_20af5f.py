# https://github.com/jamesjcai/quantum/blob/b18436492456e62db64a8b725554e07068ef2f3e/three_genes.py
from scipy.optimize import minimize

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from numpy import pi
from numpy.random import rand
import numpy as np
from numpy.linalg import norm 


N=3

r_min, r_max = 0, 2*pi
theta = r_min + rand(N) * (r_max - r_min)



def variational_circuit(qc, qr, theta):
    qc.cry(theta[0], qr[0], qr[1])
    qc.cry(theta[1], qr[1], qr[2])
    qc.cry(theta[2], qr[2], qr[0])
    return qc



#def objective(x):
#	return x[0]**2.0 + x[1]**2.0
#result = minimize(objective, theta, method='L-BFGS-B')


qreg_q = QuantumRegister(N, 'q')
creg_c = ClassicalRegister(N, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

for i in range(N):
    circuit.h(qreg_q[i])
   
circuit=variational_circuit(circuit,qreg_q,theta)

circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])

circuit.draw()

shots=10000
backend = Aer.get_backend('qasm_simulator')
job=execute(circuit, backend, shots=shots, memory=True)
result=job.result()
counts=result.get_counts(circuit)
memory = result.get_memory()

outputArray = []
for x in range(0, shots):
    converted = int(memory[x], 3)
    outputArray.append(converted)


# print(outputArray)

from qiskit.visualization import plot_histogram
plot_histogram(counts).show()


for item, _ in enumerate(counts):
    print(item)

values=[]
for execution in counts:
    values.append(counts[execution])

values = np.array(values, dtype=float)
pvalues = values / sum(values)
print(pvalues)

